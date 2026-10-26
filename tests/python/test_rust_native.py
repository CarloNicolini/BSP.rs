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
        ExecutionOptions(backend="cuda")
    with pytest.raises(ValueError):
        ExecutionOptions(threads=1025)
    with pytest.raises(ValueError):
        solve_many(formulas,[Options()])
