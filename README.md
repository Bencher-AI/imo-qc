# imo-qc

Two checks for olympiad-level math problems, as a library and a small CLI:

- **AI resistance** — a solver model attempts the problem from the statement
  alone; a separate grader model scores that attempt against your reference
  solution and rubric, out of 10.
- **Quality checks** — nine LLM reviewers judge the problem itself
  (self-containedness, internal consistency, competition scope, solvability,
  discrimination, resistance to shortcuts, wording, elegance, novelty).

The output is a diagnostic report: raw scores, per-dimension verdicts and
reasons, token usage. It deliberately does **not** produce a pass/fail decision —
thresholds are a policy choice and belong to whatever is calling this.

No deduplication, no database, no web UI. Point it at any OpenAI-compatible
endpoint.

## Install

```bash
pip install imo-qc
```

## Configure

Copy `imo-qc.example.yaml` and fill in your endpoints. Both a hosted API and a
local vLLM server work; `model` has no default so a run is never billed to a
model you did not pick.

```yaml
resistance:
  attempts: 3
  solver:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: <your-solver-model>
    max_completion_tokens: 131072
    timeout_sec: 3000
    stream: true
    extra_body:                # passed through untouched
      reasoning_effort: xhigh
  grader:                      # omitted fields inherit from solver
    model: <your-grader-model>

quality_checks:
  model:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: <your-reviewer-model>
    max_tokens: 8192
    timeout_sec: 120
```

## Use

```python
from imo_qc import QC, Problem, Solution, Rubric, load_config

qc = QC(load_config("imo-qc.yaml"))

problem = Problem(
    id="p-001",
    statement="设 n 为正整数，证明 ...",
    statement_en="Let n be a positive integer. Prove that ...",  # optional
    short_answer="42",
    subjects=["number_theory"],
    solutions=[
        Solution(
            text="Since ...",
            rubric=[                      # must sum to 10
                Rubric(score=4, criterion="reduces to the key inequality"),
                Rubric(score=6, criterion="completes the induction"),
            ],
        ),
    ],
)

report = qc.evaluate(problem)              # or: await qc.aevaluate(problem)

report.resistance.attempts[0].points       # 7
report.resistance.attempts[0].grader_raw   # the grader's reasoning
report.quality_checks["consistency"].groups[0].verdict   # "pass" | "fail" | None
report.usage.solver.total_tokens
report.model_dump()                        # JSON-serialisable
```

Each capability can run alone:

```python
qc.resistance(problem)
qc.quality_checks(problem, checks=["self_contained", "consistency"])
```

`evaluate` is synchronous and refuses to run inside an existing event loop —
inside a notebook or a web handler, await `aevaluate` instead.

### CLI

```bash
imo-qc check-config imo-qc.yaml
imo-qc run problems.jsonl -o results.jsonl --concurrency 8
```

`problems.jsonl` holds one problem per line in the same shape as `Problem`;
`results.jsonl` holds one report per line. There is no resume: a run that dies
starts over.

## What the numbers mean

**Resistance.** Each attempt is one solver call followed by one grader call. The
solver sees *only* the statement — never the reference solution, the short answer
or the rubric. The grader scores 0–10 against your rubric, which is why the
rubric must sum to 10: the grader prompt and the score parser are both fixed at
that scale. A problem with no rubric is reported as `skipped`, not judged.

The system this was extracted from treated **10/10 as the only score meaning
"the AI solved it"** — a 9 still counted as resisted. If you plan to threshold at
7 or 8 instead, your numbers will not be comparable to that convention. The tool
itself takes no position.

**Quality checks.** Each dimension answers pass/fail with a short reason.
`status` and `verdict` are separate on purpose: `status="error"` with
`verdict=None` means the call never produced a usable answer, which is not the
same as the model judging the problem defective. Dimensions whose inputs are
missing come back as `skipped`.

Three dimensions judge one solution group at a time (`competition_scope`,
`solvability`, `discrimination`); the rest judge the statement. All of them
return a `groups` list, so the shape never varies.

Two caveats on the last two dimensions:

- **elegance** is the most subjective one; its own prompt tells the model to pass
  when in doubt.
- **novelty** asks whether the problem is a reskin of something known, and
  requires the model to name what it duplicates. It depends entirely on what the
  model happens to remember. **It is not a deduplication check** and should not
  be used as one.

Prompts live in `src/imo_qc/prompts/` as plain text files. The nine quality
prompts are in Chinese; replace them if you want another language. All untrusted
content is wrapped in per-run nonce-tagged boundaries, and each prompt tells the
model to ignore instructions found inside those boundaries.

## Cost

A resistance attempt with a reasoning model and a large token budget is
expensive, and the default is three attempts per problem. `early_stop_at: 10`
cancels the remaining attempts once one reaches full marks — but note that this
only saves anything on problems the solver actually solves. Problems that resist
run every attempt either way.

`concurrency` bounds problems in flight; `max_inflight_calls` bounds LLM calls
overall. You want both — a single problem issues many calls at once.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
