"""Native Rust tests, named separately from the existing C++ binding suite."""
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import time

import numpy as np
import pytest

from pybsp_rust import Formula, Options, Solver, solve
from pybsp_rust import ExecutionOptions, solve_many
from pybsp_rust import bsp_rust as core


def test_defaults_match_rust_and_reject_invalid_inputs():
    assert asdict(Options()) == json.loads(core.default_options())
    for kwargs in [dict(damping=float("nan")), dict(seed=-1), dict(max_sweeps=0), dict(backtrack_ratio=1), dict(max_steps=True)]:
        with pytest.raises(ValueError):
            Options(**kwargs)
    with pytest.raises(ValueError):
        Formula([[0]], n=1)
    with pytest.raises(ValueError):
        core.Solver(Formula([[1]], n=1), '{"unknown":true}')


def test_solve_in_memory_has_owned_numpy_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = Formula.random(k=3, alpha=2.5, n=80, seed=3)
    result = solve(f, Options(seed=3))
    assert result.status == "sat"
    assert f.is_satisfied(result.assignment)
    assert result.assignment.dtype == np.int8
    assert result.partial_assignment.shape == (80,)
    assert not list(tmp_path.iterdir())


def test_state_control_and_array_ownership():
    f = Formula([[1, 2, 3], [-1, 2, 4], [1, -3, -4]], n=4)
    s = Solver(f)
    old = s.messages
    assert old.dtype == np.float64
    s.fix(0, True)
    assert s.assignment[0] == 1
    assert s.active_clauses == 1
    assert s.residual_formula().clauses == [[2, 4]]
    s.release(0)
    np.testing.assert_array_equal(s.messages, old)
    s.set_messages(np.zeros(f.edges))
    assert s.sweep() == 0
    report = s.converge()
    assert report["paramagnetic"]
    assert s.surveys.shape == (4, 3)
    assert s.advance()["action"] == "walksat"
    result = s.solve()
    assert result.status == "sat"
    del s
    gc.collect()
    assert np.isfinite(old).all()
    assert f.is_satisfied(result.assignment)


def test_cancellation_token_is_independent_of_solver_borrow():
    # An unsatisfiable formula with no units; WalkSAT cannot finish successfully.
    f = Formula([[1, 2], [1, -2], [-1, 2], [-1, -2]], n=2)
    s = Solver(f, Options(max_flips=100_000_000))
    s.set_messages(np.zeros(f.edges))
    token = s.cancellation_token
    with ThreadPoolExecutor(1) as executor:
        task = executor.submit(s.solve)
        time.sleep(.01)
        token.cancel()
        result = task.result(timeout=5)
    assert result.status == "cancelled"
    assert result.flips > 0
    assert result.assignment is None


def test_independent_solvers_release_gil_and_are_reproducible():
    f = Formula.random(3, 2.8, 128, seed=19)
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda _: solve(f), range(8)))
    for r in results:
        assert r.status == "sat"
        np.testing.assert_array_equal(r.assignment, results[0].assignment)
        assert r.flips == results[0].flips


def test_contradiction_is_reported_without_claiming_unsat():
    result = solve(Formula([[1], [-1]], n=1))
    assert result.status == "contradiction"
    assert result.assignment is None
    assert solve(Formula([], n=0)).assignment.shape == (0,)


def test_native_bsp_backtracking_returns_a_verified_assignment():
    f = Formula.random(k=3, alpha=4.15, n=400, seed=3)
    result = solve(f, Options(seed=3, max_flips=100_000))
    assert result.status == "sat"
    assert result.decimations > 0 and result.releases > 0
    assert f.is_satisfied(result.assignment)


def test_parallel_execution_batch_order_and_profile(tmp_path):
    f = Formula.random(3, 3.8, 80, seed=5)
    a = Solver(f)
    b = Solver(f, execution=ExecutionOptions(backend="cpu-parallel",threads=2))
    assert a.sweep_many(5) == b.sweep_many(5)
    np.testing.assert_array_equal(a.messages,b.messages)
    formulas = [Formula.random(3,2.5,80,seed=s) for s in range(4)]
    results = solve_many(formulas,[Options(seed=s) for s in range(4)],execution=ExecutionOptions(backend="cpu-parallel",threads=2))
    assert [r.seed for r in results] == list(range(4))
    assert all(f.is_satisfied(r.assignment) for f,r in zip(formulas,results))
    profile = tmp_path / "profile.json"
    solve(formulas[0],execution=ExecutionOptions(profile=profile))
    assert json.loads(profile.read_text())["execution"]["phases"]["cavities"]["calls"] > 0
    with pytest.raises(ValueError):
        ExecutionOptions(backend="unknown-backend")
    with pytest.raises(ValueError):
        ExecutionOptions(threads=1025)
    with pytest.raises(ValueError):
        solve_many(formulas,[Options()])


def test_direct_keywords_match_legacy_configuration(tmp_path):
    formula = Formula.random(k=3, alpha=2.5, n=80, seed=3)
    algorithm = dict(seed=3, backtrack_ratio=0.5, batch_fraction=0.1,
                     epsilon=0.02, damping=0.2, max_sweeps=200, max_steps=1000,
                     max_flips=100_000, walksat_restarts=2, walksat_noise=0.4,
                     timeout_seconds=None)
    hardware = dict(backend="cpu-parallel", threads=2, profile=None)
    direct = Solver(formula=formula, **algorithm, **hardware)
    legacy = Solver(formula, Options(**algorithm), execution=ExecutionOptions(**hardware))
    assert direct.options == legacy.options
    assert direct.execution == legacy.execution
    assert direct.sweep_many(5) == legacy.sweep_many(5)
    np.testing.assert_array_equal(direct.messages, legacy.messages)
    result = solve(formula=formula, **algorithm, **hardware)
    expected = solve(formula, Options(**algorithm), execution=ExecutionOptions(**hardware))
    assert result.status == expected.status == "sat"
    assert result.seed == 3
    np.testing.assert_array_equal(result.assignment, expected.assignment)
    assert (result.sp_sweeps, result.flips) == (expected.sp_sweeps, expected.flips)
    assert formula.is_satisfied(result.assignment)
    profile = tmp_path / "direct-profile.json"
    solve(formula=formula, seed=3, profile=profile)
    assert json.loads(profile.read_text())["execution"]["phases"]["cavities"]["calls"] > 0


def test_batch_direct_keywords_and_per_job_options():
    formulas = [Formula([[1], [-2]], n=2)] * 3
    results = solve_many(formulas=formulas, seed=7, backend="cpu-parallel", threads=2)
    assert [r.seed for r in results] == [7, 7, 7]
    assert all(f.is_satisfied(r.assignment) for f, r in zip(formulas, results))
    results = solve_many(formulas=formulas, options=[Options(seed=s) for s in range(3)],
                         backend="cpu", threads=2)
    assert [r.seed for r in results] == [0, 1, 2]
    assert solve_many(formulas=[], seed=3, threads=2) == []
    assert Solver(formula=formulas[0], options=Options(seed=8), backend="cpu").options.seed == 8
    assert Solver(formula=formulas[0], seed=9, execution=ExecutionOptions()).options.seed == 9


@pytest.mark.parametrize("kwargs, error, match", [
    ({"sead": 3}, TypeError, "unexpected keyword argument"),
    ({"options": Options(), "seed": 1}, TypeError, "either options"),
    ({"execution": ExecutionOptions(), "backend": "cpu"}, TypeError, "either execution"),
    ({"execution": ExecutionOptions(), "profile": None}, TypeError, "either execution"),
    ({"seed": -1}, ValueError, None),
    ({"max_sweeps": 0}, ValueError, None),
    ({"damping": float("nan")}, ValueError, None),
    ({"timeout_seconds": 0}, ValueError, None),
    ({"threads": 1025}, ValueError, None),
    ({"backend": "typo"}, ValueError, None),
    ({"execution": "cpu"}, TypeError, "execution must be"),
])
def test_direct_keywords_reject_invalid_or_ambiguous_configuration(kwargs, error, match):
    formula = Formula([[1]], n=1)
    for call, arguments in [(Solver, {"formula": formula}), (solve, {"formula": formula}),
                            (solve_many, {"formulas": [formula]})]:
        with pytest.raises(error, match=match):
            call(**arguments, **kwargs)


def test_batch_direct_keywords_reject_shared_profile(tmp_path):
    with pytest.raises(ValueError, match="profil"):
        solve_many(formulas=[Formula([[1]], n=1)], profile=tmp_path / "profile.json")
