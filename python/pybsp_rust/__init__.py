"""Native Rust BSP. Assignments use -1=free, 0=false, and 1=true."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from . import bsp_rust as _core
from .bsp_rust import Formula, CancellationToken

__version__ = _core.__version__
__all__ = ["Formula", "Options", "ExecutionOptions", "Solver", "SolveResult", "CancellationToken", "solve", "solve_many", "from_file"]


@dataclass(frozen=True)
class ExecutionOptions:
    backend: str = "cpu"
    threads: int = 0
    profile: str | None = None

    def __post_init__(self):
        _core.validate_execution(self._json())

    def _json(self):
        data = asdict(self)
        if data["profile"] is not None:
            data["profile"] = str(data["profile"])
        return json.dumps(data, allow_nan=False)


@dataclass(frozen=True)
class Options:
    seed: int = 1
    backtrack_ratio: float = 0.9
    batch_fraction: float = 0.00125
    epsilon: float = 0.01
    damping: float = 0.0
    max_sweeps: int = 1024
    max_steps: int = 100_000
    max_flips: int = 10_000_000
    walksat_restarts: int = 4
    walksat_noise: float = 0.5
    timeout_seconds: float | None = None

    def __post_init__(self):
        _core.validate_options(self._json())

    def _json(self):
        return json.dumps(asdict(self), allow_nan=False)


class SolveResult:
    def __init__(self, data):
        self.__dict__.update(data)
        self.assignment = None if data["assignment"] is None else np.asarray(data["assignment"], dtype=np.int8)
        self.partial_assignment = np.asarray(data["partial_assignment"], dtype=np.int8)

    def __repr__(self):
        return f"SolveResult(status={self.status!r}, sp_sweeps={self.sp_sweeps}, flips={self.flips})"


class Solver:
    """Independent solver state. Long native calls release Python's GIL.

    Array properties return owned snapshots. Message order follows normalized
    Formula.clauses: duplicates collapse, tautologies disappear, literals sort
    by variable and sign. Read surveys after converge().
    """
    def __init__(self, formula, options=None, *, execution=None):
        if not isinstance(formula, Formula):
            # Accept the existing pybsp.Formula without loading its C++ solver.
            formula = Formula(formula.clauses, n=formula.n)
        self.formula = formula
        self.options = Options() if options is None else options
        if not isinstance(self.options, Options):
            raise TypeError("options must be pybsp_rust.Options")
        self.execution = ExecutionOptions() if execution is None else execution
        if not isinstance(self.execution, ExecutionOptions):
            raise TypeError("execution must be pybsp_rust.ExecutionOptions")
        self._native = _core.Solver(formula, self.options._json(), self.execution._json())
        self.cancellation_token = self._native.cancellation_token

    @property
    def assignment(self):
        return np.asarray(self._native.assignment, dtype=np.int8)

    @property
    def execution_report(self):
        return json.loads(self._native.execution_report)

    @property
    def messages(self):
        return np.asarray(self._native.messages, dtype=np.float64)

    @property
    def surveys(self):
        return np.asarray(self._native.surveys, dtype=np.float64).reshape(-1, 3)

    @property
    def free_variables(self):
        return self._native.free_variables

    @property
    def active_clauses(self):
        return self._native.active_clauses

    def set_messages(self, values):
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError("messages must be a flat array in normalized clause order")
        self._native.set_messages(np.ascontiguousarray(a))

    def fix(self, variable, value):
        self._native.fix(variable, value)

    def release(self, variable):
        self._native.release(variable)

    def propagate(self):
        return self._native.propagate()

    def sweep(self):
        return self._native.sweep()

    def sweep_many(self, count):
        return self._native.sweep_many(count)

    def converge(self):
        return json.loads(self._native.converge())

    def advance(self):
        return json.loads(self._native.advance())

    def decimate(self):
        return self._native.decimate()

    def backtrack(self):
        return self._native.backtrack()

    def residual_formula(self):
        return self._native.residual_formula()

    def solve(self):
        return SolveResult(json.loads(self._native.solve()))


def solve(formula, options=None, *, execution=None):
    return Solver(formula, options, execution=execution).solve()


def solve_many(formulas, options=None, *, execution=None):
    """Solve independent jobs with bounded workers, returning results in input order."""
    formulas = list(formulas)
    formulas = [f if isinstance(f, Formula) else Formula(f.clauses, n=f.n) for f in formulas]
    if options is None:
        options = [Options()] * len(formulas)
    elif isinstance(options, Options):
        options = [options] * len(formulas)
    else:
        options = list(options)
    if len(options) != len(formulas) or any(not isinstance(o, Options) for o in options):
        raise ValueError("one Options value is required per formula")
    execution = ExecutionOptions() if execution is None else execution
    if not isinstance(execution, ExecutionOptions):
        raise TypeError("execution must be pybsp_rust.ExecutionOptions")
    return [SolveResult(data) for data in json.loads(_core.solve_many(formulas, [o._json() for o in options], execution._json()))]


def from_file(path):
    return Formula.from_dimacs(Path(path).read_text())
