"""Command line interface: JSONL in, JSONL out."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config, load_config
from .models import Problem
from .qc import QC
from .registry import ALL_CHECKS


def _read_problems(path: Path) -> list[Problem]:
    problems = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                problems.append(Problem.model_validate_json(line))
            except Exception as e:
                raise click.ClickException(f"{path}:{lineno}: {e}") from e
    return problems


async def _run_all(
    qc: QC, problems: list[Problem], checks: Optional[list[str]], concurrency: int, out
) -> int:
    gate = asyncio.Semaphore(concurrency)
    failures = 0
    write_lock = asyncio.Lock()

    async def one(problem: Problem) -> None:
        nonlocal failures
        async with gate:
            try:
                report = await qc.aevaluate(problem, checks=checks)
                payload = report.model_dump()
            except Exception as e:  # keep going; one bad problem is not fatal
                failures += 1
                payload = {"problem_id": problem.id, "error": f"{type(e).__name__}: {e}"}
        async with write_lock:
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()

    await asyncio.gather(*(one(p) for p in problems))
    return failures


@click.group()
@click.version_option(package_name="imo-qc")
def main() -> None:
    """AI-resistance and quality checks for olympiad math problems."""


@main.command()
def init() -> None:
    """Print a starter configuration to stdout: imo-qc init > imo-qc.yaml"""
    click.echo(
        (Path(__file__).parent / "example_config.yaml").read_text(encoding="utf-8"), nl=False
    )


@main.command()
@click.argument("problems", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="imo-qc.yaml",
    show_default=True,
)
@click.option("--concurrency", type=int, default=None, help="Problems in flight.")
@click.option(
    "--checks",
    default=None,
    help=f"Comma-separated subset of: {', '.join(ALL_CHECKS)}",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Only evaluate the first N problems. Use this to price a run before committing to it.",
)
@click.option(
    "--only",
    type=click.Choice(["resistance", "quality"]),
    default=None,
    help="Run one capability only. Resistance is the expensive half.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing output file.")
def run(
    problems: Path,
    output: Path,
    config_path: Path,
    concurrency: Optional[int],
    checks: Optional[str],
    limit: Optional[int],
    only: Optional[str],
    force: bool,
) -> None:
    """Evaluate every problem in a JSONL file."""
    if output.exists() and not force:
        # There is no resume, so silently truncating would destroy results that
        # already cost real money.
        raise click.ClickException(
            f"{output} already exists. Move it aside or choose another path; "
            f"pass --force to overwrite."
        )
    config = load_config(config_path)
    if only == "resistance":
        config = config.model_copy(update={"quality_checks": None})
    elif only == "quality":
        config = config.model_copy(update={"resistance": None})
    selected = [c.strip() for c in checks.split(",")] if checks else None
    problem_list = _read_problems(problems)
    if limit is not None:
        problem_list = problem_list[:limit]
    qc = QC(config)

    async def go() -> int:
        try:
            with open(output, "w", encoding="utf-8") as out:
                return await _run_all(
                    qc,
                    problem_list,
                    selected,
                    concurrency or config.concurrency,
                    out,
                )
        finally:
            await qc.aclose()

    failures = asyncio.run(go())
    click.echo(f"wrote {len(problem_list)} report(s) to {output}", err=True)
    if failures:
        click.echo(f"{failures} problem(s) failed outright", err=True)
        sys.exit(1)


@main.command("check-config")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="imo-qc.yaml",
)
@click.option("--probe/--no-probe", default=True, help="Send one tiny request per endpoint.")
def check_config(config_path: Path, probe: bool) -> None:
    """Validate a config file, and optionally try each endpoint once."""
    config: Config = load_config(config_path)
    click.echo("config OK")
    for label, model_cfg in (
        ("resistance.solver", config.resistance.solver if config.resistance else None),
        ("resistance.grader", config.resistance.grader if config.resistance else None),
        (
            "quality_checks.model",
            config.quality_checks.model if config.quality_checks else None,
        ),
    ):
        if model_cfg is None:
            continue
        click.echo(f"  {label}: {model_cfg.model} @ {model_cfg.base_url}")

    if not probe:
        return

    from .llm import LLMClient

    async def go() -> None:
        for label, model_cfg in (
            ("resistance.solver", config.resistance.solver if config.resistance else None),
            ("resistance.grader", config.resistance.grader if config.resistance else None),
            (
                "quality_checks.model",
                config.quality_checks.model if config.quality_checks else None,
            ),
        ):
            if model_cfg is None:
                continue
            client = LLMClient(
                model_cfg, http_timeout_sec=config.http_timeout_sec, retry=config.retry
            )
            try:
                text, usage = await client.complete_text("Reply with the single word OK.")
                click.echo(f"  {label}: reachable ({usage.total_tokens} tokens)")
            except Exception as e:
                raise click.ClickException(f"{label} unreachable: {e}") from e
            finally:
                await client.aclose()

    asyncio.run(go())
