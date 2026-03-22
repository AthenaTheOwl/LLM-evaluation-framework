"""HTML report generation using Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from jinja2 import Environment, FileSystemLoader

from llm_evals.models import EvalSuiteResult

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def generate_html_report(result: EvalSuiteResult, output_dir: Union[str, Path]) -> Path:
    """Generate an HTML report from eval results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html")

    html = template.render(
        result=result,
        status="PASSED" if result.passed else "FAILED",
        status_class="pass" if result.passed else "fail",
    )

    output_path = output_dir / f"{result.suite_name}_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
