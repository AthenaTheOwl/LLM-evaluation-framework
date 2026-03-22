"""Rich console output for eval results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_evals.models import DeterministicResult, EvalSuiteResult, JudgeResult, PersonaResult

console = Console()


def print_results(result: EvalSuiteResult, verbose: bool = False) -> None:
    """Print eval suite results to the console."""
    status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    header = (
        f"[bold]{result.suite_name}[/bold]  |  Model: {result.model}  |  "
        f"Run: {result.run_id}  |  {status}"
    )
    console.print(Panel(header, title="Eval Results"))

    # Summary table
    summary = Table(title="Stage Summary", show_lines=True)
    summary.add_column("Stage", style="cyan")
    summary.add_column("Score", justify="right")
    summary.add_column("Status", justify="center")

    for stage, score in result.aggregate_scores.items():
        status_icon = (
            "[green]PASS[/green]"
            if result.stage_passed.get(stage, False)
            else "[red]FAIL[/red]"
        )
        summary.add_row(stage, _format_stage_score(stage, score), status_icon)

    summary.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{result.pass_rate:.1%}[/bold] pass rate",
        "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
    )
    console.print(summary)

    # Per-case details
    if verbose:
        _print_case_details(result)

    # Regression info
    if result.regression:
        _print_regression(result)

    console.print()


def _format_stage_score(stage: str, score: float) -> str:
    if stage == "deterministic":
        return f"{score:.1%}"
    if stage in {"persona", "judge"}:
        return f"{score:.2f}/5"
    return f"{score:.3f}"


def _print_case_details(result: EvalSuiteResult) -> None:
    for cr in result.case_results:
        status = "[green]PASS[/green]" if cr.passed else "[red]FAIL[/red]"
        console.print(f"\n{status} [bold]{cr.case_id}[/bold]  ({cr.latency_ms:.0f}ms)")

        for sr in cr.stage_results:
            if isinstance(sr, DeterministicResult):
                for ar in sr.assertion_results:
                    icon = "[green]PASS[/green]" if ar.passed else "[red]FAIL[/red]"
                    console.print(f"    {icon} {ar.type.value}: {ar.message or 'OK'}")

            elif isinstance(sr, PersonaResult):
                for name, ps in sr.persona_scores.items():
                    console.print(f"    [cyan]{name}[/cyan]: {ps.overall_score:.1f}/5")
                    if ps.reasoning:
                        console.print(f"      {ps.reasoning[:120]}...")

            elif isinstance(sr, JudgeResult):
                console.print(
                    f"    [cyan]Judge[/cyan]: {sr.overall_score:.1f}/5 "
                    f"(confidence: {sr.confidence:.2f})"
                )
                if sr.calibration_delta is not None:
                    console.print(f"      Calibration delta: {sr.calibration_delta:+.2f}")
                if sr.reasoning:
                    console.print(f"      {sr.reasoning[:200]}...")


def _print_regression(result: EvalSuiteResult) -> None:
    reg = result.regression
    if reg is None:
        return

    table = Table(title="Regression Report", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Delta", justify="right")
    table.add_column("Status", justify="center")

    for metric, delta in reg.score_deltas.items():
        color = "green" if delta >= 0 else "red"
        status = "[green]PASS[/green]" if metric not in reg.regressions else "[red]v[/red]"
        table.add_row(metric, f"[{color}]{delta:+.3f}[/{color}]", status)

    console.print(table)
