"""Tests for YAML config loading and validation."""


import pytest
import yaml

from llm_evals.config import discover_suites, load_suite, validate_suite


@pytest.fixture
def temp_suite(tmp_path):
    """Create a minimal valid suite in a temp directory."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    case = {
        "id": "test-001",
        "prompt": "What is 2+2?",
        "assertions": [{"type": "contains", "value": "4"}],
    }
    with open(cases_dir / "basic.yaml", "w") as f:
        yaml.dump(case, f)

    suite = {
        "name": "test-suite",
        "model": "mock",
        "provider": "mock",
        "stages": ["deterministic"],
        "cases": ["cases/basic.yaml"],
    }
    with open(tmp_path / "suite.yaml", "w") as f:
        yaml.dump(suite, f)

    return tmp_path


class TestLoadSuite:
    def test_load_from_directory(self, temp_suite):
        suite = load_suite(temp_suite)
        assert suite.name == "test-suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "test-001"

    def test_load_from_file(self, temp_suite):
        suite = load_suite(temp_suite / "suite.yaml")
        assert suite.name == "test-suite"

    def test_inline_cases(self, tmp_path):
        suite_data = {
            "name": "inline-test",
            "model": "mock",
            "cases": [
                {"id": "inline-001", "prompt": "Hello"},
            ],
        }
        with open(tmp_path / "suite.yaml", "w") as f:
            yaml.dump(suite_data, f)

        suite = load_suite(tmp_path)
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "inline-001"

    def test_missing_suite_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_suite(tmp_path)

    def test_missing_case_file(self, tmp_path):
        suite_data = {
            "name": "broken",
            "model": "mock",
            "cases": ["cases/nonexistent.yaml"],
        }
        with open(tmp_path / "suite.yaml", "w") as f:
            yaml.dump(suite_data, f)

        with pytest.raises(FileNotFoundError):
            load_suite(tmp_path)


class TestValidateSuite:
    def test_valid_suite(self, temp_suite):
        issues = validate_suite(temp_suite)
        assert issues == []

    def test_no_cases(self, tmp_path):
        suite_data = {"name": "empty", "model": "mock", "cases": []}
        with open(tmp_path / "suite.yaml", "w") as f:
            yaml.dump(suite_data, f)
        issues = validate_suite(tmp_path)
        assert any("no test cases" in i.lower() for i in issues)

    def test_persona_stage_without_personas(self, tmp_path):
        suite_data = {
            "name": "bad-persona",
            "model": "mock",
            "stages": ["persona"],
            "cases": [{"id": "t1", "prompt": "hi"}],
        }
        with open(tmp_path / "suite.yaml", "w") as f:
            yaml.dump(suite_data, f)
        issues = validate_suite(tmp_path)
        assert any("persona" in i.lower() for i in issues)

    def test_judge_stage_without_config(self, tmp_path):
        suite_data = {
            "name": "bad-judge",
            "model": "mock",
            "stages": ["judge"],
            "cases": [{"id": "t1", "prompt": "hi"}],
        }
        with open(tmp_path / "suite.yaml", "w") as f:
            yaml.dump(suite_data, f)
        issues = validate_suite(tmp_path)
        assert any("judge" in i.lower() for i in issues)


class TestDiscoverSuites:
    def test_finds_suites(self, temp_suite):
        suites = discover_suites(temp_suite.parent)
        assert any(str(temp_suite) in str(s) for s in suites)
