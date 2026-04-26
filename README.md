<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

# N° 02 · llm-evals

> *quality is more than one score.*

a lightweight LLM evaluation framework that refuses to collapse "is this output any good?" into a single number. instead, three lenses — pointed at the same output, asked different questions, and scored independently.

`python` · `typer` · `pyyaml` · `pydantic` · `anthropic` · `openai` · `MIT` · 2024 · **status: running**

```bash
pip install -e ".[dev]"
llm-evals run eval_suites/customer_support/ --stage all --output html
```

ships with **6 suites · 46 test cases · 53 unit tests** across healthcare, food delivery, ad campaigns, content recommendation, customer support, and code review.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the three lenses

| lens | the question | LLM calls? | speed |
|---|---|---|---|
| **deterministic** | did it follow the basic rules? (format, length, required content) | no | fast |
| **persona**       | how does it land for the people it's actually for? (domain expert, end user, safety reviewer) | yes | medium |
| **judge**         | what does a stronger model think, with reasoning, confidence, and a calibrated baseline? | yes | medium |

the lenses are independent. each one takes `(case, model_output)` and returns scores. the runner aggregates, applies thresholds, and exits 1 if anything trips. `fail_fast` skips later stages if the deterministic ones fail — most of the time, that's all you need.

## the pipeline

```
YAML config
    │
    ▼
  runner ──▶ get model output
    │             │
    ├─▶ deterministic
    ├─▶ persona (multiple raters, weighted rubrics)
    └─▶ judge (CoT, confidence, human baseline)
            │
            ▼
        results ──▶ thresholds ──▶ exit 0 / exit 1
            │
            └──▶ baseline diff ──▶ regression detection
```

## a real suite, in YAML

suites live in `eval_suites/`. cases are inline or referenced from separate files. nothing custom — vanilla pyyaml, no constructors.

```yaml
name: customer-support-eval
model: claude-sonnet-4-20250514
provider: anthropic

stages: [deterministic, persona, judge]
fail_fast: true

thresholds:
  deterministic_pass_rate: 0.9
  persona_min_score: 3.5
  judge_min_score: 4.0
  regression_tolerance: 0.05

personas:
  - name: domain_expert
    description: |
      You are a senior customer support manager with 10+ years of experience.
    weight: 1.5
    rubric:
      - { name: accuracy,   weight: 2.0 }
      - { name: resolution, weight: 1.5 }

  - name: safety_reviewer
    weight: 2.0
    rubric:
      - { name: pii_safety, weight: 3.0 }

judge:
  model: claude-sonnet-4-20250514
  require_chain_of_thought: true
  confidence_threshold: 0.7
  human_baseline:
    faithfulness: 4.5

cases:
  - cases/refund_request.yaml
  - cases/escalation.yaml
```

## the assertion vocabulary

| type | what it checks |
|---|---|
| `exact_match`  | output matches exactly (whitespace trimmed) |
| `contains`     | substring is present (case-insensitive) |
| `not_contains` | substring is absent |
| `regex`        | matches a regex pattern |
| `json_schema`  | valid JSON matching a schema |
| `min_length` · `max_length` | bounds the response |
| `starts_with`  | prefix matches |

## the CLI

```
llm-evals run <suite>       run a suite through the pipeline
llm-evals validate <suite>  validate the YAML without running it
llm-evals list              list available suites
llm-evals init <name>       scaffold a new suite
llm-evals report <json>     turn saved results into HTML
```

key flags for `run`: `--stage`, `--output`, `--provider`, `--save-baseline`, `--compare-baseline`, `--fail-on-regression`, `--concurrency N`, `-v`.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## CI/CD

the included GitHub Actions workflow runs three jobs on every PR that touches `src/` or `eval_suites/`:

1. **deterministic gate** — fast pass/fail
2. **full eval** — all 3 stages, posted as a PR comment
3. **unit tests** — framework regression coverage

set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` as repo secrets.

## the suites

- **customer support** — refunds, escalation, multilingual; personas: domain expert, end user, safety reviewer
- **code reviewer** — SQL injection, naming, off-by-one; personas: senior engineer, junior dev
- **healthcare** — chest pain triage, suicidal ideation, medication interactions
- **food delivery** — tracking, allergy safety, ETA delays, refunds
- **ad campaign** — FTC/FDA compliance, targeting ethics, budget analysis
- **content recommendation** — filter bubbles, parental controls, wellbeing nudges

## design decisions

| decision | why |
|---|---|
| stages are independent, not chained | each is testable in isolation; the signals are orthogonal |
| YAML with file references | no custom constructors needed |
| sync API + ThreadPoolExecutor | right-sized for 10s–100s of cases |
| baselines as JSON in the repo | regression detection in CI without external storage |
| no database | JSON results are the data store |

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the floorplan

```
src/llm_evals/
  cli.py              typer entry point
  config.py           YAML loading + validation
  models.py           pydantic data models
  runner.py           pipeline orchestrator
  providers/          anthropic, openai, mock
  stages/
    deterministic.py
    persona.py
    judge.py
  reporting/          console, json, html, regression

eval_suites/          YAML configs
baselines/            stored baselines for regression checks
tests/                53 unit tests
.github/workflows/    CI/CD
```

## colophon

a product-minded answer to "is this LLM output any good?" — separating hard constraints, stakeholder experience, and expert judgment so each one can fail (or pass) on its own terms.

`MIT` license. *built downstairs.* — [the basement, room 7](https://github.com/AthenaTheOwl)
