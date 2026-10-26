"""Python interface to native Backtracking Survey Propagation (BSP).

Use :func:`solve` for one search, :func:`solve_many` for independent jobs, or
:class:`Solver` to inspect and control message passing. Signed CNF literals are
one-based; solver variable indices are zero-based. Assignment arrays encode
``-1`` = free, ``0`` = false, and ``1`` = true. Every SAT result is verified
against the original normalized formula. Other outcomes are heuristic failures
or limits, and do not certify that the input is unsatisfiable.
"""
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
    """Hardware and profiling settings, separate from algorithm parameters.

    Parameters
    ----------
    backend : {'cpu', 'cpu-parallel', 'metal', 'cuda', 'cuda-f32'}, default='cpu'
        Scalar CPU double precision, multicore CPU double precision, Apple GPU
        single precision, NVIDIA GPU double precision, or NVIDIA GPU single
        precision. Metal requires a macOS ``metal`` build; CUDA requires a Linux
        ``cuda`` build and a supported device. Availability is checked when a
        solver is constructed. Single precision may change the search path.
    threads : int, default=0
        Worker limit in ``0..1024``. Zero selects available logical CPUs.
        A single scalar CPU or GPU solver uses one host execution thread;
        ``solve_many`` uses this budget for independent jobs. The parallel CPU
        backend also uses it for sweeps within a job when appropriate.
    profile : str or pathlib.Path or None, default=None
        JSON profile destination written by ``Solver.solve()``. Enables phase
        timers and GPU timestamps; GPU instrumentation changes submission costs.
        Batch solving rejects profiling: profile individual solvers instead.

    Raises
    ------
    ValueError
        If the backend name or worker limit is invalid.

    Notes
    -----
    The object is immutable. The requested backend can differ from the actual
    backend after numerical recovery; inspect ``Solver.execution_report``.
    """
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
    """Immutable BSP, convergence, and WalkSAT parameters.

    Parameters
    ----------
    seed : int, default=1
        Unsigned 64-bit solver RNG seed. Controls initial messages and local
        search. It does not change a formula already generated with another seed.
    backtrack_ratio : float, default=0.9
        Target ratio of backtracking actions to decimation actions, in ``[0,1)``.
        This is a deterministic scheduling ratio, not a per-step probability.
        Zero disables scheduled backtracking. Large values increase revisiting.
    batch_fraction : float, default=0.00125
        Fraction in ``(0,1]`` of currently free variables selected per decimation.
        The batch is ``max(1, floor(batch_fraction * free_variables))``, capped by
        the number free. Smaller batches require more SP fixed points.
    epsilon : float, default=0.01
        Absolute tolerance in ``(0,1)`` on the largest *undamped* message update.
        Convergence does not establish the accuracy of the survey approximation.
    damping : float, default=0.0
        Old-message weight in ``[0,1)``: new = raw + damping * (old - raw).
        Zero is undamped. Increasing it can stabilize oscillations but slows
        relaxation. The convergence test still compares raw against old.
    max_sweeps : int, default=1024
        Positive sweep limit for each SP fixed-point attempt.
    max_steps : int, default=100000
        Positive limit on recorded BSP actions. A step can contain many sweeps.
    max_flips : int, default=10000000
        Nonnegative unsigned 64-bit WalkSAT flip budget across all restarts.
        Zero permits checking an initial completion but permits no flips.
    walksat_restarts : int, default=4
        Positive maximum number of randomized completions. Each restart has a
        quota of ``ceil(max_flips / walksat_restarts)``; the total cap still holds.
    walksat_noise : float, default=0.5
        Probability in ``[0,1]`` of a uniformly random free variable in a sampled
        unsatisfied clause when every candidate has positive break count.
        Otherwise choose among minimum-break candidates, breaking ties randomly.
    timeout_seconds : float or None, default=None
        Positive finite cooperative wall-time limit since solver construction,
        including backend setup and time between calls. None disables it.

    Raises
    ------
    ValueError
        If a probability, tolerance, limit, or integer range is invalid.

    Notes
    -----
    Values are validated by the same Rust code used by the CLI. Use
    ``dataclasses.replace(options, seed=...)`` to derive a modified configuration.
    Integer bounds also depend on the native platform's index width.
    """
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
    """Owned result record returned by :func:`solve` or :meth:`Solver.solve`.

    Attributes
    ----------
    status : str
        ``sat``, ``contradiction``, ``non_converged``, ``negative_complexity``,
        ``numerical_failure``, ``step_limit``, ``flip_limit``, ``cancelled``, or
        ``timeout``. Only ``sat`` certifies a solution; none is an UNSAT proof.
    assignment : numpy.ndarray or None
        Verified complete assignment of shape ``(n,)`` and dtype ``int8`` on
        success, containing only 0/1. None for every other outcome.
    partial_assignment : numpy.ndarray
        BSP state of shape ``(n,)``, dtype ``int8``, using -1/0/1. WalkSAT may
        complete free variables without fixing them in this partial state.
    seed : int
        Solver seed, independent of the formula-generation seed.
    wall_seconds : float
        Elapsed wall time since this solver was constructed, including setup.
    sp_sweeps, flips : int
        Cumulative completed SP sweeps and local-search flips.
    decimations, releases, unit_assignments : int
        Cumulative variable events; a variable can be fixed and released again.
    steps : list of dict
        Action records with index, action, free_variables, active_clauses,
        cumulative sp_sweeps, residual, complexity, and changed variable count.
    execution : dict
        Actual backend, precision, device, worker count, counters, memory,
        fallback reasons, device/profile errors, and optional phase timings.

    Notes
    -----
    Records are produced by the solver; applications normally do not construct
    this class directly. Arrays own their data and remain valid independently
    of the solver.
    """
    def __init__(self, data):
        self.__dict__.update(data)
        self.assignment = None if data["assignment"] is None else np.asarray(data["assignment"], dtype=np.int8)
        self.partial_assignment = np.asarray(data["partial_assignment"], dtype=np.int8)

    def __repr__(self):
        return f"SolveResult(status={self.status!r}, sp_sweeps={self.sp_sweeps}, flips={self.flips})"


class Solver:
    """Independent mutable BSP state with inspectable message passing.

    Parameters
    ----------
    formula : Formula
        Immutable normalized formula. A compatible object exposing ``clauses``
        and ``n`` is copied into a native Formula on construction.
    options : Options or None, default=None
        Algorithm settings. None creates default options.
    execution : ExecutionOptions or None, default=None
        Hardware settings. None selects scalar CPU double precision.

    Attributes
    ----------
    formula : Formula
        Normalized input formula with shared immutable native topology.
    options : Options
        Algorithm configuration supplied at construction.
    execution : ExecutionOptions
        Requested hardware configuration; actual execution is in the report.
    cancellation_token : CancellationToken
        One-way handle safe to call from another thread during computation.

    Notes
    -----
    Formula topology can be shared, but each solver has its own RNG, messages,
    assignments, counters, and cancellation token. Long native compute calls
    release Python's GIL. Use independent solvers for concurrent computation;
    do not mutate one solver concurrently from multiple threads.
    Reassigning stored Python configuration attributes does not reconfigure
    native state; construct another solver to apply new options.

    All array properties are owned snapshots. Edges follow normalized clause
    order: repeated literals collapse, tautological clauses disappear, and
    literals sort by variable then sign. Read free-variable surveys only after
    a successful ``converge()`` and before another update or graph mutation.
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
        """Owned partial assignment array with shape ``(n,)``, dtype ``int8``, -1/0/1."""
        return np.asarray(self._native.assignment, dtype=np.int8)

    @property
    def execution_report(self):
        """Fresh dict of backend, precision, memory, counters, errors, and timers."""
        return json.loads(self._native.execution_report)

    @property
    def messages(self):
        """Owned ``float64`` array of clause-to-variable warnings, shape ``(edges,)``.

        Values lie in [0,1] and follow ``formula.clauses`` flattened in order.
        Inactive/fixed incidences are zero. GPU f32 values are widened on export;
        the array dtype does not imply f64 internal arithmetic.
        """
        return np.asarray(self._native.messages, dtype=np.float64)

    @property
    def surveys(self):
        """Owned ``float64`` array, shape ``(n,3)``, columns True/False/free.

        These are normalized cluster-forcing surveys, not ordinary probabilities
        under a uniform distribution on satisfying assignments. Free-variable
        rows are meaningful after successful convergence; sweeps and mutations
        can leave them stale. Fixed rows are one-hot True or False.
        """
        return np.asarray(self._native.surveys, dtype=np.float64).reshape(-1, 3)

    @property
    def free_variables(self):
        """Number of variables currently unfixed in the BSP partial assignment."""
        return self._native.free_variables

    @property
    def active_clauses(self):
        """Number of clauses not yet satisfied by the partial assignment."""
        return self._native.active_clauses

    def set_messages(self, values):
        """Replace messages and saved values in normalized incidence order.

        Parameters
        ----------
        values : array_like
            A flat array of length formula.edges with finite probabilities in
            [0,1]. Values on currently inactive or fixed
            incidences are saved for restoration, while their live values are
            zero. Unsafe GPU f32 conversion triggers a recorded CPU fallback.

        Raises
        ------
        ValueError
            If shape, length, or probability values are invalid.

        Notes
        -----
        This invalidates surveys and clears terminal state, but does not reset
        counters, elapsed time, or cancellation. Mutating ``messages`` alone
        does not change the solver; pass the modified array here.
        """
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError("messages must be a flat array in normalized clause order")
        self._native.set_messages(np.ascontiguousarray(a))

    def fix(self, variable, value):
        """Fix a zero-based variable to a Boolean value and simplify incidences.

        The variable must currently be free; already fixed or out-of-range
        requests raise ValueError. Unit propagation is deferred until
        ``propagate()``, ``converge()``, or ``advance()``. Free surveys become
        stale. This explicit decision can later be released.
        """
        self._native.fix(variable, value)

    def release(self, variable):
        """Unfix a zero-based decision variable and restore saved edge messages.

        Already free, out-of-range, and unit-forced variables raise ValueError.
        Release clears terminal
        state and invalidates free-variable surveys.
        """
        self._native.release(variable)

    def propagate(self):
        """Apply the queued unit implications and return newly forced count.

        Raises RuntimeError on a contradictory residual, cancellation, timeout,
        or numerical/device failure. A contradiction under heuristic decisions
        is not a proof that the original formula is UNSAT.
        """
        return self._native.propagate()

    def sweep(self):
        """Run one synchronous cavity/clause update and return undamped residual.

        Does not propagate units or refresh surveys. Raises RuntimeError on
        numerical failure, contradiction, cancellation, or timeout. Intended
        for controlled experiments; use ``converge()`` for fixed points.
        """
        return self._native.sweep()

    def sweep_many(self, count):
        """Run ``count`` raw synchronous sweeps, synchronizing snapshots once.

        Returns the final undamped residual; count=0 returns 0.0. Units and
        surveys are not refreshed. This amortizes GPU readback costs compared
        with repeated ``sweep()`` calls. Numerical recovery and cooperative
        cancellation/timeout remain enabled. Native failures raise RuntimeError.
        """
        return self._native.sweep_many(count)

    def converge(self):
        """Propagate units, find an SP fixed point, then refresh surveys.

        Returns a dict with ``sweeps`` for this attempt, ``residual`` of the last
        undamped update, total ``complexity`` estimate, and ``paramagnetic``.
        The latter is an implementation stopping diagnostic, not a SAT verdict.
        Raises RuntimeError on nonconvergence, contradiction, numerical failure,
        cancellation, or timeout. At most ``options.max_sweeps`` are attempted
        per precision attempt; GPU recovery can retry in CPU f64.
        """
        return json.loads(self._native.converge())

    def advance(self):
        """Converge and perform one scheduled BSP action, returning a step dict.

        Actions are ``decimate``, ``backtrack``, or ``walksat``. The WalkSAT
        action may finish the search. Terminal conditions raise RuntimeError;
        call ``solve()`` to retrieve a terminal result record instead. Step
        free/active counts describe the state before the action.
        """
        return json.loads(self._native.advance())

    def decimate(self):
        """Fix the top survey-certainty batch and return the number changed.

        Call ``converge()`` first: this method uses cached surveys and does not
        compute a fixed point. Ties use ascending variable index; an equal
        True/False forcing survey selects False. Does not propagate units.
        """
        return self._native.decimate()

    def backtrack(self):
        """Release low-score stored decisions and return the number changed.

        The release budget uses the last decimation batch plus pending unit
        implications, capped by eligible decisions. Scores are frozen when a
        variable is fixed; unit-forced variables are not eligible. Does not
        recompute surveys or propagate units.
        """
        return self._native.backtrack()

    def residual_formula(self):
        """Return active clauses with fixed literals removed, keeping original n.

        Variable IDs are not compacted, so unused variables remain declared.
        This formula encodes the remaining constraints; keep the partial
        assignment separately to reconstruct a completion of the original CNF.
        """
        return self._native.residual_formula()

    def solve(self):
        """Continue until a terminal condition and return a :class:`SolveResult`.

        Operational search outcomes are recorded in ``result.status`` rather
        than raised as RuntimeError. Existing work, counters, elapsed time,
        decisions, and cancellation are retained. Every SAT assignment passes
        CPU verification. A profile-write error is recorded in execution
        metadata without replacing an otherwise valid result.
        """
        return SolveResult(json.loads(self._native.solve()))


def solve(formula, options=None, *, execution=None):
    """Construct a fresh :class:`Solver` and return its terminal result.

    Parameters are the same as ``Solver``. Input/configuration errors raise
    ValueError or TypeError; heuristic failures are represented by status.

    Examples
    --------
    >>> result = solve(Formula([[1], [-2]], n=2))
    >>> result.status
    'sat'
    >>> result.assignment.tolist()
    [1, 0]
    """
    return Solver(formula, options, execution=execution).solve()


def solve_many(formulas, options=None, *, execution=None):
    """Solve independent jobs with bounded workers, preserving input order.

    Parameters
    ----------
    formulas : iterable of Formula
        Inputs, materialized before solving. Repeated native Formula objects
        share immutable CPU topology while keeping independent solver state.
    options : Options or iterable of Options or None, default=None
        One immutable configuration for all jobs, one per job, or defaults.
        Assign distinct seeds explicitly when requesting independent restarts.
    execution : ExecutionOptions or None, default=None
        Shared worker/backend settings. GPU concurrency is additionally bounded
        by estimated device memory. Profiling must be None for batch calls.

    Returns
    -------
    list of SolveResult
        One result per input, in original order, including heuristic failures.

    Raises
    ------
    ValueError
        If lengths/configurations disagree, profiling is requested, or backend
        setup fails.
    """
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
    """Read a DIMACS CNF file into a normalized :class:`Formula`.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with exactly one ``p cnf N M`` header and zero-terminated clauses.

    Returns
    -------
    Formula
        Immutable formula preserving original variable IDs and declared n.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If DIMACS counts, termination, or literal ranges are invalid.
    """
    return Formula.from_dimacs(Path(path).read_text())
