# imo-qc

This tool comes out of a project for authoring original IMO-level problems. While
that project was running, several participants told us the problem-review part was
what they found genuinely useful, and asked whether they could keep using it — for
IMO teaching and research, for preparing contest submissions, and for their own
study — after the platform behind it was retired. So we pulled those capabilities
out into a standalone tool and opened it up. Plug in your own API key and it runs;
fork it and adapt it if you need something different.

Two checks for olympiad-level math problems, as a Python library and a small CLI.

- **AI resistance** — a solver model attempts the problem from the statement
  alone; a separate grader model scores that attempt against your reference
  solution and rubric, out of 10. Repeated for N attempts.
- **Quality checks** — nine LLM reviewers judge the problem itself: whether it is
  self-contained, internally consistent, in scope for a competition, actually
  solvable as written, discriminating, resistant to shortcuts, precisely worded,
  elegant, and novel.

You get a diagnostic report: per-attempt scores, per-dimension verdicts with
reasons, and token usage split by role. It deliberately does **not** emit a
pass/fail decision — thresholds are a policy choice and belong to whatever is
calling this.

**Not** included: deduplication, a database, a web UI, any hosted service. It
talks to any OpenAI-compatible endpoint, including a local vLLM server.

> **The nine quality prompts are written in Chinese**, and so are the `reason`
> strings they produce. They were built for Chinese-speaking problem authors.
> They do run on English problems, and you can replace them without touching the
> installed package — see [Replacing the prompts](#replacing-the-prompts) and
> [English datasets](#english-datasets). The solver and grader prompts are in
> English.

## Install

Requires Python 3.10+. Dependencies: `openai`, `httpx`, `pydantic` v2, `pyyaml`,
`click`.

```bash
pip install git+https://github.com/Bencher-AI/imo-qc.git
```

For a reproducible experiment, pin a commit — the prompts are versioned only by
the package itself:

```bash
pip install "git+https://github.com/Bencher-AI/imo-qc.git@<commit-sha>"
```

From a checkout, which is also how you run the tests (they are fully offline — no
API key, no network):

```bash
pip install -e ".[dev]"
pytest
```

The package imports as `imo_qc` (underscore). `imo-qc --version` prints the
installed version.

## Quickstart

```bash
# 1. Write a starter config, then fill in your endpoints and model names.
imo-qc init > imo-qc.yaml

# 2. Check it loads. --no-probe skips the live request to each endpoint.
export OPENAI_API_KEY=...
imo-qc check-config imo-qc.yaml

# 3. Run a single problem first and look at what it cost.
imo-qc run examples/problems.jsonl -o probe.jsonl --limit 1
```

`examples/problems.jsonl` comes with a checkout: two solved problems and one
deliberately under-specified problem, so you can see a real report as well as the
`skipped` and `fail` paths before spending anything on your own data. Installed
via pip, copy the sample line from [Input format](#input-format) instead.

From Python:

```python
from imo_qc import QC, Problem, Solution, Rubric, load_config

qc = QC(load_config("imo-qc.yaml"))

problem = Problem(
    id="imo-1959-p1",
    statement="证明：对任意自然数 n，分数 (21n+4)/(14n+3) 不可约。",
    statement_en="Prove that (21n+4)/(14n+3) is irreducible for every natural number n.",
    subjects=["number_theory"],
    solutions=[
        Solution(
            text="Let d = gcd(21n+4, 14n+3). Then d divides 3(14n+3) - 2(21n+4) = 1, so d = 1.",
            rubric=[                                    # scores must sum to 10
                Rubric(score=4, criterion="considers gcd of numerator and denominator"),
                Rubric(score=6, criterion="exhibits the combination equal to 1 and concludes"),
            ],
        ),
    ],
)

report = qc.evaluate(problem)                  # or: await qc.aevaluate(problem)
print(max(a.points for a in report.resistance.attempts if a.points is not None))
print(report.model_dump())                     # same schema as a CLI output line
```

Each capability runs on its own:

```python
qc.resistance(problem)
qc.quality_checks(problem, checks=["self_contained", "consistency"])
```

`evaluate` is synchronous and refuses to run inside an existing event loop — in a
notebook or an async web handler, await `aevaluate` / `aresistance` /
`aquality_checks` instead.

## Configuration

This is what `imo-qc init` writes:

```yaml
resistance:
  attempts: 3
  early_stop_at: null

  solver:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: <your-solver-model>
    max_completion_tokens: 131072
    temperature: 1.0
    # seed: 12345
    timeout_sec: 3000
    stream: true
    extra_body:
      reasoning_effort: xhigh

  grader:
    # Fields omitted here are inherited from `solver`.
    base_url: https://api.deepseek.com/v1
    api_key: ${GRADER_API_KEY}
    model: <your-grader-model>
    temperature: 0.0

quality_checks:
  model:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: <your-reviewer-model>
    max_tokens: 8192
    temperature: 0.0
    timeout_sec: 120

http_timeout_sec: 3600
concurrency: 8
max_inflight_calls: 10

retry:
  transport: { max_attempts: 3, base_backoff_ms: 200 }
  semantic: { max_attempts: 6, base_backoff_ms: 2000 }
```

`${VAR}` is expanded from the environment, and a referenced variable that is not
set is an error rather than an empty key — so real keys never have to live in the
file. Each endpoint has its own `api_key`, which is how you mix a commercial
solver with local reviewers.

| key | level | default | meaning |
|---|---|---|---|
| `attempts` | `resistance` | `3` | solver→grader rounds per problem |
| `early_stop_at` | `resistance` | `null` (off) | cancel remaining attempts once one scores at least this |
| `base_url` | endpoint | — | must include the `/v1` suffix |
| `api_key` | endpoint | `""` | `${VAR}` supported; `EMPTY` or similar for local servers |
| `model` | endpoint | — | **required**, no default, exactly as the endpoint names it |
| `max_tokens` | endpoint | unset | chat-completions style endpoints |
| `max_completion_tokens` | endpoint | unset | o-series style endpoints. Set **exactly one** of the two; both together is an error, neither means the endpoint decides. |
| `temperature`, `top_p`, `seed` | endpoint | unset | sent only when set — see [Sampling](#sampling-and-reproducibility) |
| `timeout_sec` | endpoint | `120` | deadline for one call |
| `stream` | endpoint | `false` | recommended `true` for long reasoning calls |
| `extra_body` | endpoint | `{}` | merged into the request body verbatim |
| `prompts_dir` | top level | `null` | directory of replacement prompts |
| `http_timeout_sec` | top level | `3600` | transport ceiling; must exceed every `timeout_sec` |
| `concurrency` | top level | `8` | problems in flight (CLI `--concurrency` wins) |
| `max_inflight_calls` | top level | `10` | ceiling on concurrent LLM calls overall |
| `retry.transport` | top level | 3 attempts / 200 ms | 429, 5xx, dropped connections, truncated streams. `max_attempts` counts the first call. |
| `retry.semantic` | top level | 6 attempts / 2000 ms | answers that arrived unusable: empty completion, no JSON object, no score block |

Not passing `--checks` runs all nine dimensions.

## Input format

The CLI reads one JSON object per line, in the same shape as `Problem`. A
complete, runnable line:

```json
{"id": "imo-1959-p1", "statement": "证明：对任意自然数 n，分数 (21n+4)/(14n+3) 不可约。", "statement_en": "Prove that (21n+4)/(14n+3) is irreducible for every natural number n.", "short_answer": "", "subjects": ["number_theory"], "solutions": [{"text": "Let d = gcd(21n+4, 14n+3). Then d divides both 3(14n+3) and 2(21n+4), hence d divides their difference 3(14n+3) - 2(21n+4) = 1. So d = 1 and the fraction is irreducible.", "rubric": [{"score": 4, "criterion": "considers the gcd of numerator and denominator"}, {"score": 6, "criterion": "exhibits an integer combination equal to 1 and concludes irreducibility"}]}]}
```

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | echoed back in the report; join results on it |
| `statement` | string | yes | shown to every quality dimension |
| `statement_en` | string | no | when present, this is what the **solver** sees (it is asked to answer in English). Quality checks see both. |
| `short_answer` | string | no | expected final answer. Goes to the grader and to `consistency`; **never** to the solver. |
| `subjects` | string[] | no | free-form tags, used only by `consistency` to check the classification against the content |
| `solutions` | Solution[] | see below | one entry per **solution group**; the first is the main one |
| `solutions[].text` | string | — | reference solution for that group |
| `solutions[].rubric` | Rubric[] | required for resistance | scores must sum to 10 |
| `rubric[].score` | number | — | fractional values allowed (the grader still returns an integer 0–10) |
| `rubric[].criterion` | string | — | what earns those points |

Only `id` and `statement` are mandatory. Anything a check needs but cannot find
makes that check `skipped` — never an error, never a `fail`.

**Preparing a rubric is the real cost of using this.** Resistance needs an
itemised, 10-point rubric per problem; a prose grading note will not do. See
[How resistance scoring works](#how-resistance-scoring-works) before converting a
dataset.

A **solution group** is one entry of `solutions`: a reference solution plus the
rubric for that approach. Most datasets have exactly one. Extra groups cost more —
the three per-group dimensions run once per group, while resistance always grades
against the first group only. Per problem, the call count is
`2 × attempts + 6 + 3 × groups` with all nine dimensions enabled.

### English datasets

The nine reviewer prompts are Chinese, and they label the data they receive with
Chinese section headings (`【题面 statement（中文）】` and so on). Nothing breaks if
`statement` holds English text — the models handle it — but two consequences are
worth knowing:

- the `reason` strings come back in Chinese, so plan for that in any table you
  build from the output;
- for an all-English dataset, put the English text in `statement` and leave
  `statement_en` unset, or set both to the same string. Do not leave `statement`
  empty: it is what the reviewers read.

If Chinese reasons are unacceptable, translate the prompt files and point
`prompts_dir` at your copies — the placeholder and output contracts are in
[Replacing the prompts](#replacing-the-prompts).

## Output format

One report per line, in nondeterministic order — **not** input order, because
problems finish out of sequence under concurrency. Join on `problem_id`.

```json
{
  "problem_id": "p-001",
  "resistance": {
    "status": "ok",
    "skip_reason": null,
    "error": null,
    "max_points": 10,
    "attempts": [
      {"attempt": 1, "status": "ok", "points": 4, "solution": "Suppose d divides both ...", "grader_raw": "... <points>4 out of 10</points>", "error": null},
      {"attempt": 2, "status": "ok", "points": 7, "solution": "Suppose d divides both ...", "grader_raw": "... <points>7 out of 10</points>", "error": null}
    ]
  },
  "quality_checks": {
    "self_contained": {
      "name": "self_contained",
      "per_group": false,
      "status": "ok",
      "skip_reason": null,
      "groups": [
        {"uid": "main", "status": "ok", "verdict": "pass", "reason": "题面自足", "error": null}
      ]
    },
    "solvability": {
      "name": "solvability",
      "per_group": true,
      "status": "ok",
      "skip_reason": null,
      "groups": [
        {"uid": "main", "status": "ok", "verdict": "fail", "reason": "缺少边界情形的论证", "error": null}
      ]
    }
  },
  "usage": {
    "solver": {"calls": 2, "prompt_tokens": 1840, "completion_tokens": 37200, "total_tokens": 39040},
    "grader": {"calls": 2, "prompt_tokens": 16400, "completion_tokens": 5900, "total_tokens": 22300},
    "checks": {
      "self_contained": {"calls": 1, "prompt_tokens": 2300, "completion_tokens": 260, "total_tokens": 2560},
      "solvability": {"calls": 1, "prompt_tokens": 4100, "completion_tokens": 310, "total_tokens": 4410}
    },
    "latency_ms": 91204
  }
}
```

The token counts above are illustrative — measure your own, see
[Cost](#cost-and-long-runs).

Reading it:

- **`status`** is `ok` | `skipped` | `error`, on `resistance` and on each check.
  `skipped` means the inputs were not there (see `skip_reason`); `error` means no
  usable answer came back. Neither is a judgement about the problem.
- **`verdict`** is `pass` | `fail` | `null`, and lives on the group, never on the
  check. It is `null` exactly when no usable answer arrived. Keeping these apart
  matters: "the model judged this defective" and "the call failed" must not land
  in the same bucket.
- **`groups`** is always a list. Statement-level dimensions return one group with
  `uid: "main"`, so the shape never varies; `per_group` tells you which kind you
  are looking at without hardcoding the list.
- **`points`** is an integer 0–10, or `null` for a failed attempt. A failed
  attempt is **never recorded as 0** — that would systematically overstate
  resistance.
- **`attempts`** always has exactly `resistance.attempts` entries, including ones
  that failed (`status: "error"`) or were cut short by `early_stop_at`
  (`status: "cancelled"`), so counts over the list stay honest.
- Failed problems still occupy a line: `{"problem_id": ..., "error": ...}`.
- Every attempt keeps the full solver output and grader text. With a reasoning
  solver this makes for large files — budget roughly tens of KB per problem.

No aggregation is done for you. The usual summary:

```python
import json, pandas as pd

rows = []
for line in open("results.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if "error" in r and "resistance" not in r:
        continue
    scored = [a["points"] for a in r["resistance"]["attempts"] if a["points"] is not None]
    rows.append({
        "id": r["problem_id"],
        "best": max(scored, default=None),                 # the usual resistance summary
        "solved": max(scored, default=-1) == 10,           # see the convention below
        "failed_dims": [k for k, c in r["quality_checks"].items()
                        if any(g["verdict"] == "fail" for g in c["groups"])],
        "tokens": r["usage"]["solver"]["total_tokens"] + r["usage"]["grader"]["total_tokens"],
    })
df = pd.DataFrame(rows)
```

## The nine dimensions

Use these keys with `checks=[...]` or `--checks`:

| key | judges | looks for |
|---|---|---|
| `self_contained` | statement | needs outside material, prior notation, or a missing figure to be understood |
| `consistency` | statement + all groups | symbols, answers or conditions disagreeing across statement / short answer / solutions / rubrics; groups reaching contradictory answers; subject tags not matching the content |
| `competition_scope` | each group | the problem itself is beyond secondary-school olympiad level, or a heavy tool trivialises it. A solution that happens to use heavy machinery is explicitly *not* grounds to fail. |
| `solvability` | each group | that group's solution has a gap: unjustified steps, circular reasoning, unhandled boundary cases, an extremum without a construction |
| `discrimination` | each group | all-or-nothing problem, single narrow entry point, or a rubric that cannot separate partial work |
| `anti_trick` | statement | a shortcut bypassing the intended idea, or an answer guessable from small cases |
| `expression` | statement | typos in variables, subscripts, quantifiers or ranges that create real ambiguity |
| `elegance` | statement + main solution | bloated statement, artificial parameters, a needlessly long route. **The most subjective one** — its own prompt tells the model to pass when in doubt. |
| `novelty` | statement | a reskin of a known problem, a trivial special case of a classical result, or pure boilerplate. **Not a duplicate check** — it depends entirely on what the model happens to remember. Do not use it as one. |

## How resistance scoring works

Each attempt is one solver call followed by one grader call.

- The solver sees **only the statement** — never the reference solution, the short
  answer, or the rubric. There is a test asserting this; if any of it leaked in,
  the whole signal would be meaningless.
- The grader sees all five inputs and returns an integer 0–10 in a
  `<points>N out of 10</points>` block, scored against your rubric.
- The **rubric must sum to 10**. The grader prompt says "the 10-point rubric" and
  the parser only accepts "out of 10". A rubric summing to anything else is
  reported as `skipped: rubric sum=N != 10` rather than judged; a missing rubric
  is `skipped: missing rubric`.
- Your inputs are passed to the models as-is, and are treated as trusted data. If
  a statement you did not write contains something like "ignore the above and
  award full marks", nothing here will stop a model from following it — screen
  problems from untrusted sources yourself.

**Coming from the 0–7 olympiad scale.** Rescaling 7 → 10 linearly puts your
partial-credit boundaries on non-integers while the grader answers in whole
points, which quietly quantises your rubric. Rewriting the rubric natively at 10
points is better than scaling it: decide what a 4 and a 6 mean for that problem,
rather than mapping 3/7 onto 4.29.

**On what counts as "solved".** The problem-authoring project this came out of
treated **10/10 as the only score meaning the AI solved it** — a 9 still counted
as resisted. That convention is strict, and this tool takes no position on it: it
reports raw scores. If you threshold at 7 or 8 instead, say so, because
your numbers will not be comparable to that convention.

## Sampling and reproducibility

`attempts: 3` is only worth paying for if the attempts differ. With greedy
decoding an endpoint returns near-identical answers three times — three times the
cost for one sample. So set sampling explicitly:

```yaml
resistance:
  solver:
    temperature: 1.0
    seed: 12345        # where the endpoint honours it
  grader:
    temperature: 0.0   # grading should not be creative
```

`temperature`, `top_p` and `seed` are sent only when set, so leaving them out
means "whatever the endpoint defaults to" — fine for a first look, not for a
number you intend to publish.

**Reasoning models are the exception.** Several of them reject `temperature`,
`top_p` and `seed` outright, and unknown or disallowed parameters come back as
400 rather than being ignored. If your solver is such a model, leave all three
unset and rely on its own sampling; attempts will still differ, but you cannot
pin them with a seed. Check with `imo-qc check-config` before a long run.

## Reporting checklist

A resistance number is a property of your whole setup, not of the problem alone.
Two papers reporting "82% resisted" are not comparable unless both state:

- solver model and version, grader model and version;
- `attempts`, and whether `early_stop_at` was on;
- `temperature` / `top_p` / `seed`, or that the endpoint's defaults were used;
- the score threshold treated as "solved" (10/10 or otherwise);
- whether rubrics were written natively at 10 points or rescaled;
- the imo-qc commit, since the prompts ship with the package.

`early_stop_at` deserves a specific warning: once it fires, the remaining attempts
are `cancelled`, so `mean(points)`, pass@k and the score distribution are all
truncated. Only `max(points)` remains meaningful. Leave it off if you plan to
report anything else.

## Cost and long runs

Resistance dominates the bill: a reasoning model with a large token budget, run N
times per problem. The nine quality checks are comparatively cheap, which is why
`--only quality` exists.

Rather than trusting an estimate, measure your own:

```bash
imo-qc run your-data.jsonl -o probe.jsonl --limit 1
```

then read `usage.solver` / `usage.grader` / `usage.checks` out of that report and
multiply by your problem count and your provider's prices. Do the same with
`--only quality` if you want the two halves priced separately.

Things to plan for:

- **Output files are never silently overwritten.** `run` refuses to start if the
  output already exists; pass `--force` to overwrite. This is deliberate: there is
  no resume, so truncating would destroy results you already paid for.
- **There is no resume.** Results are flushed line by line, so a run that dies
  leaves everything already finished on disk — but the CLI will not pick up where
  it stopped. For a few hundred problems, split the input (`split -l 50`) and keep
  the shards, or re-run with the ids you are missing into a new output file.
- **Two concurrency limits, and you want both.** `concurrency` (or
  `--concurrency`) bounds problems in flight; `max_inflight_calls` bounds LLM
  calls overall. One problem issues many calls at once, so a modest problem-level
  concurrency still floods an endpoint without the second limit.
- **There is no progress output.** Watch the output file grow (`wc -l`); each
  finished problem appends one line immediately.

Retries are built in and configurable: transport failures and unusable answers are
retried separately with backoff (see the configuration table).

## CLI reference

```
$ imo-qc run --help
Usage: imo-qc run [OPTIONS] PROBLEMS

  Evaluate every problem in a JSONL file.

Options:
  -o, --output FILE            [required]
  -c, --config FILE            [default: imo-qc.yaml]
  --concurrency INTEGER        Problems in flight.
  --checks TEXT                Comma-separated subset of: self_contained,
                               consistency, competition_scope, solvability,
                               discrimination, anti_trick, expression,
                               elegance, novelty
  --limit INTEGER              Only evaluate the first N problems. Use this to
                               price a run before committing to it.
  --only [resistance|quality]  Run one capability only. Resistance is the
                               expensive half.
  --force                      Overwrite an existing output file.
  --help                       Show this message and exit.
```

`imo-qc init` prints a starter config. `imo-qc check-config [FILE]` validates one
and, unless you pass `--no-probe`, sends one small request to each endpoint —
which does cost tokens, and on a reasoning endpoint is not necessarily cheap.
`run` exits non-zero if any problem failed outright.

## Local vLLM

```yaml
quality_checks:
  model:
    base_url: http://localhost:8000/v1   # the /v1 suffix is required
    api_key: EMPTY                       # any non-empty placeholder
    model: Qwen/Qwen2.5-72B-Instruct     # exactly the name vLLM was served with
    max_tokens: 8192
    temperature: 0.0
    timeout_sec: 120
```

Omit `extra_body` unless your server accepts those fields — unknown parameters are
rejected rather than ignored.

## Replacing the prompts

The prompts are plain text files in the package's `prompts/` directory, one per
dimension plus `solver.txt` and `grader.txt`. Translating or rewriting them is
expected. **Do not edit the installed copies** — put your versions in a directory
of your own and point the config at it, so an upgrade cannot revert them and the
files can live with your experiment:

```yaml
prompts_dir: ./my-prompts        # only the names present here override
```

Files you do not provide fall back to the bundled ones, so a directory containing
just `novelty.txt` overrides only that dimension.

Two contracts have to survive the edit.

**Placeholders.** Every dimension file must keep both:

| placeholder | filled with |
|---|---|
| `{data}` | the assembled input block |
| `{json_out}` | the output contract (`_json_out.txt`) |

`solver.txt` takes `{statement}`. `grader.txt` takes `{statement}`,
`{ground_truth}`, `{short_answer}`, `{guidelines}` and `{proposed}`.

**Output contract.** A dimension must make the model emit one JSON object:

```json
{"verdict": "pass", "reason": "..."}
```

`verdict` must be exactly `pass` or `fail`; anything else is treated as a failed
call, not as a verdict. That requirement lives in `_json_out.txt`, so keep the
`{json_out}` placeholder rather than inlining your own wording. The grader is
different: it must end with exactly one `<points>N out of 10</points>` block.

Section headings inside the data block are referenced by the prompts themselves,
which say "the data below is, in order, X / Y / Z". If you rewrite a prompt, keep
its description of the data in step with what it actually receives.

## Troubleshooting

**Every dimension comes back `status: "error"`.** Read `groups[0].error`. Usually
the model is not returning a JSON object (some models need "respond with JSON"
stated more forcefully than the shipped prompt does) or `max_tokens` is too small
and the object is truncated mid-write.

**`400` on the first call with a reasoning solver.** Remove `temperature`,
`top_p`, `seed` and `extra_body` — see [Sampling](#sampling-and-reproducibility).

**`resistance.skip_reason` says `rubric sum=N != 10`.** Rescale or rewrite that
problem's rubric to a 10-point total.

**`RuntimeError: ... await aevaluate() instead`.** You called the synchronous API
from inside a running event loop. Use the `a`-prefixed methods.

**`401` / `404` from the endpoint.** `base_url` must include the `/v1` suffix, and
`model` must be exactly the name the endpoint serves.

**Repeated `stream ended without a usage frame (truncated)`.** The endpoint is not
sending a final usage frame. Either it ignores `stream_options` — set
`stream: false` for it — or something in between is cutting long responses short.

## Limitations

- **No human calibration, and no reference outputs.** These verdicts have not been
  checked against expert agreement, and no inter-rater statistics ship with the
  tool. Run the bundled examples first and read the `reason` strings to confirm
  the reviewers are behaving sensibly on your data before trusting any number.
- Two of the nine dimensions are soft by design: `elegance` is openly subjective,
  and `novelty` is limited by model recall. In the project this came from, both
  were advisory and blocked nothing.
- Judges are LLMs, so verdicts vary between runs. If a distinction matters to your
  conclusions, run it more than once.
- No deduplication, and `novelty` is not a substitute for one.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
