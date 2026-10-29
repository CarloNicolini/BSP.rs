"""Public SDK contracts independent of native numerical implementation details."""
import inspect
import re

import pytest

from pybsp_rust import (
    ExecutionOptions, Formula, Options, Solver, SolveResult, from_file,
    solve, solve_many,
)


def test_explicit_keyword_signatures_and_individual_numpy_parameter_docs():
    settings = set(Options.__dataclass_fields__) | set(ExecutionOptions.__dataclass_fields__)
    for call in (Solver, solve, solve_many):
        signature = inspect.signature(call)
        assert settings <= signature.parameters.keys()
        assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        for name in settings:
            assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
    objects = [Formula, Options, ExecutionOptions, Solver, SolveResult, solve, solve_many, from_file]
    for cls in (Solver, Formula):
        objects.extend(getattr(cls, name) for name in dir(cls)
                       if not name.startswith("_") and callable(getattr(cls, name)))
    for call in objects:
        signature = inspect.signature(call)
        documentation = inspect.getdoc(call) or ""
        if inspect.isclass(call):
            documentation += "\n" + (inspect.getdoc(call.__init__) or "")
        for name in signature.parameters:
            if name != "self":
                assert re.search(rf"(?m)^\s*{name} : .+", documentation), (call, name)


def test_direct_path_does_not_construct_configuration_dataclasses(monkeypatch):
    def unexpected_record(self):
        raise AssertionError("direct calls must not construct a configuration record")

    monkeypatch.setattr(Options, "__post_init__", unexpected_record)
    monkeypatch.setattr(ExecutionOptions, "__post_init__", unexpected_record)
    formula = Formula([[1], [-2]], n=2)
    assert Solver(formula=formula, seed=3, backend="cpu").solve().status == "sat"
    assert solve(formula=formula, seed=4).seed == 4
    assert [r.seed for r in solve_many(formulas=[formula] * 2, seeds=[5, 6])] == [5, 6]


def test_compatibility_records_are_optional_and_preserve_identity(tmp_path):
    formula = Formula([[1]], n=1)
    options = Options(seed=7)
    execution = ExecutionOptions(profile=tmp_path / "profile.json")
    solver = Solver(formula=formula, options=options, execution=execution)
    assert solver.options is options
    assert solver.execution is execution
    direct = Solver(formula=formula, seed=7, profile=tmp_path / "direct.json")
    assert direct.options == options
    assert direct.execution.profile == tmp_path / "direct.json"


def test_per_job_seeds_preserve_order_and_allow_shared_settings():
    formula = Formula.random(k=3, alpha=2.5, n=80, seed=3)
    seeds = [7, 2, 9]
    results = solve_many(formulas=(formula for _ in seeds), seeds=iter(seeds),
                         damping=0.2, backend="cpu-parallel", threads=2)
    assert [r.seed for r in results] == seeds
    for seed, result in zip(seeds, results):
        expected = solve(formula=formula, seed=seed, damping=0.2)
        assert formula.is_satisfied(result.assignment)
        assert result.assignment.tolist() == expected.assignment.tolist()
    assert solve_many(formulas=[], seeds=[]) == []


@pytest.mark.parametrize("settings,error", [
    ({"seeds": []}, ValueError),
    ({"seeds": [1, 2]}, ValueError),
    ({"seeds": [-1]}, ValueError),
    ({"seeds": [2**64]}, ValueError),
    ({"seeds": [True]}, ValueError),
    ({"seeds": [None]}, ValueError),
    ({"seeds": [1], "seed": 1}, TypeError),
    ({"seeds": [1], "options": Options()}, TypeError),
])
def test_per_job_seeds_reject_invalid_or_ambiguous_inputs(settings, error):
    with pytest.raises(error):
        solve_many(formulas=[Formula([[1]], n=1)], **settings)


@pytest.mark.parametrize("settings", [{"seed": -1}, {"seed": None},
                                      {"threads": 1025}, {"damping": float("nan")}])
def test_empty_batches_still_validate_shared_settings(settings):
    with pytest.raises(ValueError):
        solve_many(formulas=[], **settings)
