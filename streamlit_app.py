"""Streamlit viewer for the checked-in eval suites."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml


ROOT = Path(__file__).parent
SUITES_DIR = ROOT / "eval_suites"


def load_suites() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite_path in sorted(SUITES_DIR.glob("*/suite.yaml")):
        with suite_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        case_refs = data.get("cases", []) or []
        rows.append(
            {
                "suite": suite_path.parent.name,
                "name": data.get("name", suite_path.parent.name),
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
                "stages": ", ".join(data.get("stages", []) or []),
                "cases": len(case_refs),
                "path": str(suite_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return rows


st.set_page_config(page_title="LLM eval suite browser", layout="wide")
st.title("LLM eval suite browser")
st.caption(
    "A read-only viewer over the repo's deterministic, persona, and judge eval suites."
)

suites = load_suites()

left, right = st.columns(2)
left.metric("eval suites", len(suites))
right.metric("test cases", sum(int(row["cases"]) for row in suites))

st.dataframe(suites, width="stretch", hide_index=True)

selected = st.selectbox("suite", [str(row["suite"]) for row in suites])
suite_path = SUITES_DIR / selected / "suite.yaml"

with suite_path.open("r", encoding="utf-8") as handle:
    suite = yaml.safe_load(handle) or {}

st.subheader(str(suite.get("name", selected)))
st.write("Stages:", ", ".join(suite.get("stages", []) or []))
st.write("Thresholds:")
st.json(suite.get("thresholds", {}))

st.subheader("Run locally")
st.code(
    f"python -m pip install -e .\nllm-evals validate {suite_path.as_posix()}\n"
    f"llm-evals run {suite_path.as_posix()} --stage deterministic --output console",
    language="bash",
)
