"""Prevent documentation call signatures from drifting away from the SDK."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_doc_examples", Path(__file__).resolve().parents[2] / "tools/check_doc_examples.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_documented_python_calls_bind_to_current_api():
    python_count, rust_count = checker.check_documents()
    assert python_count > 0 and rust_count > 0


def test_checker_rejects_unknown_names_and_missing_required_parameters():
    for source in ['solve(formula=None, sead=3)', 'Solver(seed=3)',
                   'Formula.random(k=3, alpha=2.5)',
                   'solver = Solver(formula=None)\nsolver.removed_method()',
                   'formula = Formula([], n=0)\nformula.is_satisfied(unknown=[])']:
        with pytest.raises(ValueError):
            checker.check_python(source, "regression")


def test_checker_extracts_python_inside_shell_heredocs(tmp_path):
    page = tmp_path / "example.md"
    page.write_text("```sh\nuv run --no-sync python - <<'PY'\nsolve(formula=None, sead=3)\nPY\n```\n")
    blocks = list(checker.snippets(page))
    assert len(blocks) == 1 and blocks[0][0] == "python"
    with pytest.raises(ValueError):
        checker.check_python(blocks[0][2], str(page))
