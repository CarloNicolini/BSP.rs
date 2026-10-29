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
from functools import cached_property
import json
from pathlib import Path

import numpy as np
from . import bsp_rust as _core
from .bsp_rust import Formula, CancellationToken

__version__ = _core.__version__
__all__ = ["Formula", "Options", "ExecutionOptions", "Solver", "SolveResult", "CancellationToken", "solve", "solve_many", "from_file"]


@dataclass(frozen=True)
class ExecutionOptions:
    """Optional compatibility record for hardware and profiling settings.

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
    Prefer direct backend, threads, and profile keywords in new code.
    The object is immutable. The requested backend can differ from the actual
    backend after numerical recovery; inspect ``Solver.execution_report``.
    """
    backend: str = "cpu"
    threads: int = 0
    profile: str | Path | None = None

    def __post_init__(self):
        _core.validate_execution(self._json())

    def _json(self):
        data = asdict(self)
        if data["profile"] is not None:
            data["profile"] = str(data["profile"])
        return json.dumps(data, allow_nan=False)


@dataclass(frozen=True)
class Options:
    """Optional immutable compatibility record for algorithm parameters.

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
    Prefer direct algorithm keywords in new code.
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


class _Default:
    """Mark an omitted keyword so explicit defaults still detect conflicts."""

    def __repr__(self):
        return "DEFAULT"


_DEFAULT = _Default()
_ALGORITHM_DEFAULTS = json.loads(_core.default_options())
_HARDWARE_DEFAULTS = {"backend": "cpu", "threads": 0, "profile": None}


def _direct_settings(values):
    """Collect explicitly supplied solver keywords from a public call."""
    return {name: value for name, value in values.items()
            if (name in _ALGORITHM_DEFAULTS or name in _HARDWARE_DEFAULTS)
            and value is not _DEFAULT}


def _configuration(options, execution, parameters):
    """Resolve compatibility objects to the dictionaries sent to Rust."""
    algorithm = {k: v for k, v in parameters.items() if k in _ALGORITHM_DEFAULTS}
    hardware = {k: v for k, v in parameters.items() if k in _HARDWARE_DEFAULTS}
    if options is not None:
        if algorithm:
            raise TypeError("use either options or direct algorithm keywords, not both")
        if not isinstance(options, Options):
            raise TypeError("options must be pybsp_rust.Options")
        algorithm = asdict(options)
    else:
        algorithm = _ALGORITHM_DEFAULTS | algorithm
    if execution is not None:
        if hardware:
            raise TypeError("use either execution or direct hardware keywords, not both")
        if not isinstance(execution, ExecutionOptions):
            raise TypeError("execution must be pybsp_rust.ExecutionOptions")
        hardware = asdict(execution)
    else:
        hardware = _HARDWARE_DEFAULTS | hardware
    return algorithm, hardware


def _json(settings):
    """Serialize native settings without accepting non-finite JSON numbers."""
    if settings.get("profile") is not None:
        settings = settings | {"profile": str(settings["profile"])}
    return json.dumps(settings, allow_nan=False)


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
        """Convert a native result record into owned Python arrays.

        Parameters
        ----------
        data : dict
            Decoded native result containing status, assignments, counters,
            action history, and execution metadata.
        """
        self.__dict__.update(data)
        self.assignment = None if data["assignment"] is None else np.asarray(data["assignment"], dtype=np.int8)
        self.partial_assignment = np.asarray(data["partial_assignment"], dtype=np.int8)

    def __repr__(self):
        """Return a concise status and work-counter representation."""
        return f"SolveResult(status={self.status!r}, sp_sweeps={self.sp_sweeps}, flips={self.flips})"


class Solver:
    """Independent mutable BSP state with inspectable message passing.

    Parameters
    ----------
    formula : Formula
        Immutable normalized formula. Compatible objects exposing clauses and n
        are copied into a native Formula on construction.
    options : Options or None, default=None
        Optional compatibility configuration. Prefer the named algorithm keywords.
        Combining this object with any algorithm keyword raises TypeError.
    execution : ExecutionOptions or None, default=None
        Optional compatibility configuration. Prefer backend, threads, and profile.
        Combining this object with any hardware keyword raises TypeError.
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

    Attributes
    ----------
    formula : Formula
        Normalized input with shared immutable native topology.
    cancellation_token : CancellationToken
        Thread-safe one-way handle for cooperative interruption.

    Raises
    ------
    ValueError
        If a setting, formula, or backend is invalid or unavailable.
    TypeError
        If a keyword is unknown or configuration sources conflict.

    Notes
    -----
    The DEFAULT marker in the signature means omitted; effective defaults are
    listed for each parameter above. Direct calls send dictionaries to Rust without
    constructing configuration dataclasses. Existing Options calls remain supported.
    Each solver owns its RNG, assignments, messages, counters, and cancellation
    state. Native calls release the GIL; do not mutate a solver concurrently.
    Reassigning a Python configuration attribute does not reconfigure native state.

    Array properties are owned snapshots in normalized clause incidence order.
    Only read free-variable surveys after successful converge() and before another
    update or graph mutation. These are cluster-forcing surveys, not ordinary SAT
    marginals. The solver is a heuristic and cannot prove UNSAT.

    Examples
    --------
    >>> formula = Formula([[1], [-2]], n=2)
    >>> Solver(formula=formula, seed=3).solve().status
    'sat'
    """
    def __init__(
        self, formula, options=None,
        *,
        execution=None,
        seed=_DEFAULT,
        backtrack_ratio=_DEFAULT,
        batch_fraction=_DEFAULT,
        epsilon=_DEFAULT,
        damping=_DEFAULT,
        max_sweeps=_DEFAULT,
        max_steps=_DEFAULT,
        max_flips=_DEFAULT,
        walksat_restarts=_DEFAULT,
        walksat_noise=_DEFAULT,
        timeout_seconds=_DEFAULT,
        backend=_DEFAULT,
        threads=_DEFAULT,
        profile=_DEFAULT,
    ):
        """Initialize native solver state using the class parameter contract."""
        algorithm, hardware = _configuration(options, execution, _direct_settings(locals()))
        if not isinstance(formula, Formula):
            formula = Formula(formula.clauses, n=formula.n)
        self.formula = formula
        self._algorithm = algorithm
        self._hardware = hardware
        if options is not None:
            self.options = options
        if execution is not None:
            self.execution = execution
        self._native = _core.Solver(formula, _json(algorithm), _json(hardware))
        self.cancellation_token = self._native.cancellation_token

    @cached_property
    def options(self):
        """Return a lazily materialized algorithm compatibility snapshot.

        Returns
        -------
        Options
            Immutable settings snapshot; reading it does not reconfigure Rust.
        """
        return Options(**self._algorithm)

    @cached_property
    def execution(self):
        """Return a lazily materialized hardware compatibility snapshot.

        Returns
        -------
        ExecutionOptions
            Requested hardware settings. Inspect execution_report for actual use.
        """
        return ExecutionOptions(**self._hardware)

    @property
    def assignment(self):
        """Return an owned snapshot of the current partial assignment.

        Returns
        -------
        numpy.ndarray
            Shape (n,), dtype int8, with -1 for free, 0 for false, and 1 for true.
            Mutating this snapshot does not change native state.
        """
        return np.asarray(self._native.assignment, dtype=np.int8)

    @property
    def execution_report(self):
        """Return current backend, resource, fallback, and profiling metadata.

        Returns
        -------
        dict
            Fresh report with requested/actual backend, precision, device, memory,
            counters, fallbacks, errors, and enabled phase timings.
        """
        return json.loads(self._native.execution_report)

    @property
    def messages(self):
        """Return an owned snapshot of clause-to-variable warnings.

        Returns
        -------
        numpy.ndarray
            Shape (formula.edges,), dtype float64, probabilities in [0, 1], in
            normalized clause incidence order. Fixed/inactive incidences are zero.

        Notes
        -----
        GPU f32 values are widened on export; float64 snapshots do not imply f64
        internal arithmetic. Pass edited values to set_messages() to update state.
        """
        return np.asarray(self._native.messages, dtype=np.float64)

    @property
    def surveys(self):
        """Return an owned snapshot of variable cluster-forcing surveys.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3), dtype float64, columns true, false, and unfrozen. Fixed rows
            are one-hot true/false. Free rows are meaningful only after converge().

        Notes
        -----
        These are not ordinary marginals of a uniform satisfying-assignment measure.
        Sweeps and graph mutations can leave free-variable surveys stale.
        """
        return np.asarray(self._native.surveys, dtype=np.float64).reshape(-1, 3)

    @property
    def free_variables(self):
        """Return the number of currently unfixed variables.

        Returns
        -------
        int
            Count of free variables in the BSP partial assignment.
        """
        return self._native.free_variables

    @property
    def active_clauses(self):
        """Return the number of clauses not yet satisfied by fixed variables.

        Returns
        -------
        int
            Count of active clauses in the residual graph.
        """
        return self._native.active_clauses

    def set_messages(self, values):
        """Replace live and frozen messages in normalized incidence order.

        Parameters
        ----------
        values : array_like
            Flat array of length formula.edges with finite probabilities in [0, 1].
            Inactive/fixed values are saved for restoration; live incidences stay zero.

        Returns
        -------
        None
            Native state is updated in place.

        Raises
        ------
        ValueError
            If the array is not flat, has the wrong length, or contains invalid values.

        Notes
        -----
        This invalidates surveys and clears terminal state. Counters, elapsed time,
        and cancellation persist. Unsafe GPU f32 conversion records a CPU fallback.
        """
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError("messages must be a flat array in normalized clause order")
        self._native.set_messages(np.ascontiguousarray(a))

    def fix(self, variable, value):
        """Fix one variable and simplify its incidences.

        Parameters
        ----------
        variable : int
            Zero-based index of a currently free variable in [0, formula.n).
        value : bool
            True fixes the variable true; False fixes it false.

        Returns
        -------
        None
            Graph state is updated in place.

        Raises
        ------
        ValueError
            If the variable is out of range or already fixed.

        Notes
        -----
        Unit propagation is deferred until propagate(), converge(), or advance().
        Free surveys become stale. This explicit decision can later be released.
        """
        self._native.fix(variable, value)

    def release(self, variable):
        """Unfix an explicit decision and restore its frozen messages.

        Parameters
        ----------
        variable : int
            Zero-based index of a fixed decision variable in [0, formula.n).

        Returns
        -------
        None
            Graph state is updated in place and terminal state is cleared.

        Raises
        ------
        ValueError
            If the index is invalid, the variable is free, or it was unit-forced.

        Notes
        -----
        Free surveys become stale. Unit implications are not rolled back; construct a
        fresh solver to explore an independent branch with fresh implications.
        """
        self._native.release(variable)

    def propagate(self):
        """Apply queued unit implications.

        Returns
        -------
        int
            Number of newly forced variables in this call.

        Raises
        ------
        RuntimeError
            On a contradictory residual, cancellation, timeout, or numerical/device
            failure. A contradiction under heuristic decisions is not an UNSAT proof.
        """
        return self._native.propagate()

    def sweep(self):
        """Run one synchronous cavity/clause update.

        Returns
        -------
        float
            Maximum absolute undamped message update in the sweep.

        Raises
        ------
        RuntimeError
            On numerical failure, contradiction, cancellation, or timeout.

        Notes
        -----
        Units are not propagated and surveys are not refreshed. Use converge() for a
        fixed-point attempt. The returned array snapshots are synchronized.
        """
        return self._native.sweep()

    def sweep_many(self, count):
        """Run raw synchronous sweeps, synchronizing snapshots once.

        Parameters
        ----------
        count : int
            Nonnegative number of sweeps. Zero returns 0.0 without updating messages.

        Returns
        -------
        float
            Final undamped residual, or 0.0 for zero sweeps.

        Raises
        ------
        OverflowError
            If count is negative or outside the native index range.
        RuntimeError
            On numerical/device failure, contradiction, cancellation, or timeout.

        Notes
        -----
        Units and surveys are not refreshed. GPU readback is amortized across sweeps;
        numerical recovery and cooperative interruption remain enabled.
        """
        return self._native.sweep_many(count)

    def converge(self):
        """Propagate units, attempt an SP fixed point, then refresh surveys.

        Returns
        -------
        dict
            sweeps for this attempt, last undamped residual, total complexity estimate,
            and paramagnetic diagnostic. The latter is not a SAT verdict.

        Raises
        ------
        RuntimeError
            On nonconvergence, contradiction, numerical/device failure, cancellation,
            or timeout.

        Notes
        -----
        At most max_sweeps are attempted per precision attempt. GPU numerical recovery
        can retry on CPU f64. Convergence alone does not certify survey accuracy.
        """
        return json.loads(self._native.converge())

    def advance(self):
        """Converge and perform one scheduled BSP action.

        Returns
        -------
        dict
            Step record with action (decimate/backtrack/walksat), index, pre-action
            free/active counts, cumulative sweeps, residual, complexity, and change count.

        Raises
        ------
        RuntimeError
            On a terminal condition, including calls after termination.

        Notes
        -----
        WalkSAT may finish the search. Use solve() to retrieve a terminal result record.
        """
        return json.loads(self._native.advance())

    def decimate(self):
        """Fix the highest-certainty free-variable batch using cached surveys.

        Returns
        -------
        int
            Number of variables changed by this decision batch.

        Raises
        ------
        RuntimeError
            If interruption or a native operational failure prevents the action.

        Notes
        -----
        Call converge() first. This method does not compute a fixed point or propagate
        units. Equal certainty uses ascending variable index; equal true/false forcing
        selects false.
        """
        return self._native.decimate()

    def backtrack(self):
        """Release eligible decisions with the lowest frozen certainty.

        Returns
        -------
        int
            Number of decisions released by this action.

        Raises
        ------
        RuntimeError
            If interruption or a native operational failure prevents the action.

        Notes
        -----
        The budget uses the preceding decimation batch plus pending unit implications,
        capped by eligible decisions. Unit-forced variables cannot be released. This
        does not recompute surveys or propagate units.
        """
        return self._native.backtrack()

    def residual_formula(self):
        """Return active clauses with fixed literals removed.

        Returns
        -------
        Formula
            Normalized residual with the original declared variable count and labels.

        Notes
        -----
        Variable IDs are not compacted. Retain the partial assignment separately when
        reconstructing a completion of the original input.
        """
        return self._native.residual_formula()

    def solve(self):
        """Continue the current search until a terminal condition.

        Returns
        -------
        SolveResult
            Owned result with a CPU-verified assignment only for status 'sat'. Other
            statuses report heuristic failures or limits and do not prove UNSAT.

        Notes
        -----
        Existing work, decisions, counters, cancellation, and elapsed time are retained.
        Operational failures are represented in status, rather than raised as
        RuntimeError. A profile-write error is recorded in execution metadata without
        replacing an otherwise valid result.
        """
        return SolveResult(json.loads(self._native.solve()))


def solve(
    formula, options=None,
    *,
    execution=None,
    seed=_DEFAULT,
    backtrack_ratio=_DEFAULT,
    batch_fraction=_DEFAULT,
    epsilon=_DEFAULT,
    damping=_DEFAULT,
    max_sweeps=_DEFAULT,
    max_steps=_DEFAULT,
    max_flips=_DEFAULT,
    walksat_restarts=_DEFAULT,
    walksat_noise=_DEFAULT,
    timeout_seconds=_DEFAULT,
    backend=_DEFAULT,
    threads=_DEFAULT,
    profile=_DEFAULT,
):
    """Construct a fresh solver and return one terminal result.

    Parameters
    ----------
    formula : Formula
        Immutable normalized formula, or a compatible object exposing clauses and n.
    options : Options or None, default=None
        Optional compatibility configuration. Prefer the named algorithm keywords.
        Combining this object with any algorithm keyword raises TypeError.
    execution : ExecutionOptions or None, default=None
        Optional compatibility configuration. Prefer backend, threads, and profile.
        Combining this object with any hardware keyword raises TypeError.
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

    Returns
    -------
    SolveResult
        Verified assignment on status 'sat'; otherwise a heuristic failure or limit.

    Raises
    ------
    ValueError
        If input/settings are invalid or backend setup fails.
    TypeError
        If keywords are unknown or configuration sources conflict.

    Notes
    -----
    DEFAULT denotes an omitted setting; effective defaults are listed above.
    Formula generation has its own seed. A failed search is not an UNSAT proof.

    Examples
    --------
    >>> result = solve(formula=Formula([[1], [-2]], n=2), seed=3)
    >>> result.status, result.assignment.tolist()
    ('sat', [1, 0])
    """
    return Solver(formula=formula, options=options, execution=execution,
                  **_direct_settings(locals())).solve()


def solve_many(
    formulas, options=None,
    *,
    execution=None,
    seed=_DEFAULT,
    backtrack_ratio=_DEFAULT,
    batch_fraction=_DEFAULT,
    epsilon=_DEFAULT,
    damping=_DEFAULT,
    max_sweeps=_DEFAULT,
    max_steps=_DEFAULT,
    max_flips=_DEFAULT,
    walksat_restarts=_DEFAULT,
    walksat_noise=_DEFAULT,
    timeout_seconds=_DEFAULT,
    backend=_DEFAULT,
    threads=_DEFAULT,
    profile=_DEFAULT,
    seeds=None,
):
    """Solve independent jobs with bounded workers and stable input order.

    Parameters
    ----------
    formulas : iterable of Formula
        Inputs, materialized before solving. Repeated Formula objects share immutable
        CPU topology while each job retains independent mutable solver state.
    options : Options or iterable of Options or None, default=None
        Optional compatibility settings, shared or one per formula. Prefer named
        algorithm keywords and seeds for the common per-job seed case.
        Combining this object with any algorithm keyword raises TypeError.
    execution : ExecutionOptions or None, default=None
        Optional compatibility configuration. Prefer backend, threads, and profile.
        Combining this object with any hardware keyword raises TypeError.
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
    seeds : iterable of int or None, default=None
        One unsigned 64-bit solver seed per formula. Length must match formulas.
        Combine with shared algorithm and hardware keywords. Do not combine with
        seed or options. None broadcasts seed (whose effective default is 1).

    Returns
    -------
    list of SolveResult
        Results in input order, including heuristic failures. Empty input returns [].

    Raises
    ------
    ValueError
        If lengths/settings disagree, profile is not None, or backend setup fails.
    TypeError
        If keywords are unknown or configuration sources conflict.

    Notes
    -----
    Shared algorithm keywords apply to every job. Use seeds for independent restart
    seeds, or compatibility Options sequences for arbitrary per-job configurations.
    GPU concurrency is additionally bounded by estimated device memory. DEFAULT
    marks omission; effective defaults are listed above. Profile individual Solver
    objects instead of using a shared batch profile path.

    Examples
    --------
    >>> formula = Formula([[1], [-2]], n=2)
    >>> results = solve_many(formulas=[formula] * 3, seeds=[3, 4, 5], threads=2)
    >>> [result.seed for result in results]
    [3, 4, 5]
    """
    parameters = _direct_settings(locals())
    if seeds is not None and (seed is not _DEFAULT or options is not None):
        raise TypeError("use seeds without seed or options")
    sequence = options is not None and not isinstance(options, Options)
    if sequence and any(name in _ALGORITHM_DEFAULTS for name in parameters):
        raise TypeError("use either options or direct algorithm keywords, not both")
    algorithm, hardware = _configuration(None if sequence else options, execution, parameters)
    formulas = [f if isinstance(f, Formula) else Formula(f.clauses, n=f.n) for f in formulas]
    if seeds is not None:
        seeds = list(seeds)
        if len(seeds) != len(formulas):
            raise ValueError("one seed is required per formula")
        configurations = [algorithm | {"seed": value} for value in seeds]
    elif sequence:
        options = list(options)
        if len(options) != len(formulas) or any(not isinstance(o, Options) for o in options):
            raise ValueError("one Options value is required per formula")
        configurations = [asdict(o) for o in options]
    else:
        configurations = [algorithm] * len(formulas)
    if not formulas:
        _core.validate_options(_json(algorithm))
    data = _core.solve_many(formulas, [_json(o) for o in configurations], _json(hardware))
    return [SolveResult(item) for item in json.loads(data)]


def from_file(path):
    """Read a DIMACS CNF file into a normalized :class:`Formula`.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 text file with one ``p cnf N M`` header and zero-terminated clauses.

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
    return Formula.from_dimacs(Path(path).read_text(encoding="utf-8"))
