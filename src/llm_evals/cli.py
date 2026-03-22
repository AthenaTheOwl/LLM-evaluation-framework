"""CLI interface for llm-evals."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console(force_terminal=True)

app = typer.Typer(
    name="llm-evals",
    help="A 3-stage LLM evaluation framework: deterministic, persona, and LLM-as-a-judge.",
    no_args_is_help=True,
)


@app.command()
def run(
    suite_path: str = typer.Argument(..., help="Path to eval suite YAML or directory"),
    stage: str = typer.Option("all", help="Stage to run: all, deterministic, persona, judge"),
    output: str = typer.Option("console", help="Output format: console, json, html"),
    output_dir: str = typer.Option("./eval-results", help="Directory for output files"),
    save_baseline: bool = typer.Option(False, "--save-baseline", help="Save results as baseline"),
    compare_baseline: bool = typer.Option(
        False, "--compare-baseline", help="Compare against baseline"
    ),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit code 1 on regression"
    ),
    provider: Optional[str] = typer.Option(
        None, help="Override provider for both the model-under-test and judge"
    ),
    model: Optional[str] = typer.Option(None, help="Override model"),
    concurrency: int = typer.Option(5, help="Parallel eval cases"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run an evaluation suite through the pipeline."""
    from llm_evals.config import load_suite
    from llm_evals.providers.base import get_provider
    from llm_evals.reporting.console import print_results
    from llm_evals.reporting.html import generate_html_report
    from llm_evals.reporting.json_report import save_json_report
    from llm_evals.reporting.regression import compare_baseline as do_compare
    from llm_evals.reporting.regression import save_baseline as do_save
    from llm_evals.runner import EvalRunner

    suite = load_suite(suite_path)
    if model:
        suite.model = model
    if provider:
        suite.provider = provider
        if suite.judge is not None:
            suite.judge.provider = provider

    stages_filter = None if stage == "all" else [stage]

    model_provider = get_provider(suite.provider)
    runner = EvalRunner(
        suite=suite,
        model_provider=model_provider,
        concurrency=concurrency,
        stages_filter=stages_filter,
    )

    console.print(f"[bold]Running eval suite:[/bold] {suite.name}")
    console.print(f"  Model: {suite.model} ({suite.provider})")
    console.print(f"  Stages: {', '.join(s.name for s in runner.stages)}")
    console.print(f"  Cases: {len(suite.cases)}")
    console.print()

    result = runner.run()

    # Baseline comparison
    if compare_baseline:
        regression = do_compare(result, tolerance=suite.thresholds.regression_tolerance)
        result.regression = regression

    # Output
    if output == "console" or output == "all":
        print_results(result, verbose=verbose)

    if output in ("json", "all"):
        path = save_json_report(result, output_dir)
        console.print(f"[dim]JSON report saved to {path}[/dim]")

    if output in ("html", "all"):
        path = generate_html_report(result, output_dir)
        console.print(f"[dim]HTML report saved to {path}[/dim]")

    # Save baseline
    if save_baseline:
        path = do_save(result)
        console.print(f"[dim]Baseline saved to {path}[/dim]")

    # Exit code
    if fail_on_regression and result.regression and not result.regression.passed:
        console.print("[red]Regression detected -- exiting with code 1[/red]")
        raise typer.Exit(code=1)

    if not result.passed:
        raise typer.Exit(code=1)


@app.command()
def validate(
    suite_path: str = typer.Argument(..., help="Path to eval suite YAML or directory"),
):
    """Validate an eval suite configuration without running it."""
    from llm_evals.config import validate_suite

    issues = validate_suite(suite_path)
    if not issues:
        console.print("[green]PASS: Suite configuration is valid[/green]")
    else:
        console.print("[red]FAIL: Validation issues found:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_suites(
    directory: str = typer.Option("eval_suites", help="Directory to search for suites"),
):
    """List available eval suites."""
    from llm_evals.config import discover_suites, load_suite

    suites = discover_suites(directory)
    if not suites:
        console.print(f"[yellow]No eval suites found in {directory}[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Available Eval Suites")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Cases", justify="right")
    table.add_column("Stages")
    table.add_column("Model")

    for suite_path in suites:
        try:
            suite = load_suite(suite_path)
            table.add_row(
                suite.name,
                str(suite_path.parent.relative_to(Path(directory))),
                str(len(suite.cases)),
                ", ".join(suite.stages),
                suite.model,
            )
        except Exception as e:
            table.add_row(str(suite_path), "", "ERR", str(e), "")

    console.print(table)


@app.command()
def init(
    name: str = typer.Argument(..., help="Name for the new eval suite"),
    directory: str = typer.Option("eval_suites", help="Parent directory"),
):
    """Scaffold a new eval suite."""
    import yaml

    suite_dir = Path(directory) / name
    cases_dir = suite_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    suite_config = {
        "name": name,
        "description": f"Eval suite for {name}",
        "version": "1.0",
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "stages": ["deterministic", "persona", "judge"],
        "fail_fast": True,
        "thresholds": {
            "deterministic_pass_rate": 0.9,
            "persona_min_score": 3.5,
            "judge_min_score": 3.5,
            "regression_tolerance": 0.05,
        },
        "personas": [
            {
                "name": "domain_expert",
                "description": (
                    "You are a domain expert evaluating this response for accuracy "
                    "and completeness."
                ),
                "weight": 1.0,
                "rubric": [
                    {
                        "name": "accuracy",
                        "description": "Is the information correct?",
                        "weight": 2.0,
                    },
                    {
                        "name": "completeness",
                        "description": "Does it address all aspects?",
                        "weight": 1.0,
                    },
                ],
            }
        ],
        "judge": {
            "model": "claude-sonnet-4-20250514",
            "provider": "anthropic",
            "require_chain_of_thought": True,
            "confidence_threshold": 0.7,
            "rubric": [
                {
                    "name": "overall_quality",
                    "description": "Holistic quality assessment",
                    "weight": 1.0,
                },
                {
                    "name": "faithfulness",
                    "description": "Is the response grounded in facts?",
                    "weight": 2.0,
                },
            ],
        },
        "cases": ["cases/example.yaml"],
    }

    example_case = {
        "id": f"{name}-example-001",
        "description": "Example test case",
        "prompt": "Hello, how can you help me?",
        "reference_output": "I can help you with a variety of tasks. How can I assist you today?",
        "assertions": [
            {"type": "contains", "value": "help"},
            {"type": "max_length", "value": 500},
        ],
        "tags": ["example"],
    }

    with open(suite_dir / "suite.yaml", "w") as f:
        yaml.dump(suite_config, f, default_flow_style=False, sort_keys=False)

    with open(cases_dir / "example.yaml", "w") as f:
        yaml.dump(example_case, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]PASS: Created eval suite at {suite_dir}[/green]")
    console.print(f"  - {suite_dir / 'suite.yaml'}")
    console.print(f"  - {cases_dir / 'example.yaml'}")


@app.command()
def report(
    results_path: str = typer.Argument(..., help="Path to results JSON file"),
    output_dir: str = typer.Option("./eval-results", help="Directory for HTML report"),
):
    """Generate an HTML report from saved JSON results."""
    from llm_evals.reporting.html import generate_html_report
    from llm_evals.reporting.json_report import load_json_report

    result = load_json_report(results_path)
    path = generate_html_report(result, output_dir)
    console.print(f"[green]PASS: HTML report generated at {path}[/green]")


if __name__ == "__main__":
    app()
