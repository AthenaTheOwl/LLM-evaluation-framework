# LLM evaluation framework

Forty-six cases sit in six eval suites. A refund answer can pass its JSON shape and still fail the safety reviewer. This runner keeps those failures separate.

## What it does

This repo is a small evaluation bench for language-model behavior. It is built around fixtures, judge prompts, and repeatable reports rather than one grand score.

It checks:

- answer quality against expected behavior
- refusal behavior on unsafe or unsupported requests
- citation and grounding claims where the fixture includes source text
- structured-output shape
- regression drift across saved report runs

The point is modest and useful: make a model answer survive the same set of questions twice.

## Try it

Install the package, then run the evaluation suite:

```bash
python -m pip install -e .
llm-evals run eval_suites --output console
```

Validate the checked-in suites without running models:

```bash
llm-evals validate eval_suites
```

Run the API-free quickstart fixture and write a JSON report:

```bash
llm-evals run examples/quickstart_eval --stage deterministic --provider mock --output json --output-dir examples/reports
```

The checked-in example report is `examples/reports/quickstart-smoke.json`.

Run each case more than once. A case that passes on the first try and fails on the second is not a case you can put in front of a customer, and a single run cannot see it:

```bash
llm-evals run examples/reliability_eval --stage deterministic --provider mock --repeat 3
```

```text
Reliability over 3 attempts
  pass@1 100%   pass@3 100%   pass^3 50%
  steady-refund-answer             PPP
  flaky-refund-window              PFP  <- not every time
```

pass@1 is the first attempt, pass@k is any attempt, and pass^k is every attempt. Add `--min-pass-hat-k 1.0` to exit 1 when any case is not reliable. The mock provider replays each case's `metadata.mock_responses` in order, one per attempt, so the fixture needs no API key.

## Live demo

This repo ships a Streamlit wrapper for the same local runner.

<!-- live-url -->

Streamlit entrypoint:

```text
streamlit_app.py
```

Local run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Repo shape

```text
llm_eval_framework/
  cli.py
  runner.py
  schemas.py
  scoring.py
fixtures/
  suites/
  cases/
reports/
tests/
streamlit_app.py
```

## Why this exists

Most model demos make failure feel anecdotal. This repo makes it boring enough to repeat.

Each case has a fixture. Each run writes a report. The report can be checked in, compared, and used as evidence when a prompt, model, or scoring rule changes.

## Development

Run tests:

```bash
python -m pytest
```

Run formatting or linting only if the local checkout has those tools installed. The test suite is the contract for this repo.

## License

MIT.
