use crate::{CancellationToken, ExecutionOptions, Formula, Options, SolveJob, Solver};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn runtime_error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
fn json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(runtime_error)
}
fn options(text: Option<&str>) -> PyResult<Options> {
    let options = match text {
        Some(s) => serde_json::from_str::<Options>(s).map_err(value_error)?,
        None => Options::default(),
    };
    options.validate().map_err(value_error)?;
    Ok(options)
}

#[pyclass(name = "Formula", module = "pybsp_rust.bsp_rust", frozen)]
#[derive(Clone)]
/// Immutable normalized CNF formula with one-based signed literals.
///
/// Parameters
/// ----------
/// clauses : sequence of sequences of int
///     Each inner sequence is a disjunction. Positive i means x_i, negative
///     -i means not x_i. Zero is invalid here (it is a DIMACS terminator only).
/// n : int or None, default=None
///     Declared variable count. None infers the largest absolute literal,
///     or zero for an empty formula. Explicit n retains unused variables.
///
/// Notes
/// -----
/// Repeated literals collapse, literals sort by variable/sign, and tautological
/// clauses disappear. Repeated whole clauses and empty clauses remain. Copies
/// share immutable Rust topology. Input errors raise ValueError; integer
/// conversion outside native ranges can raise OverflowError.
struct PyFormula {
    inner: Formula,
}
#[pymethods]
impl PyFormula {
    #[new]
    #[pyo3(signature = (clauses, n=None))]
    /// Construct an immutable normalized formula.
    ///
    /// Parameters
    /// ----------
    /// clauses : sequence of sequences of int
    ///     One-based signed literals; zero is invalid. Repeated literals collapse
    ///     and tautological clauses disappear. Empty clauses remain.
    /// n : int or None, default=None
    ///     Declared variable count, retaining unused variables. None infers the
    ///     largest absolute literal, or zero for no literals.
    ///
    /// Returns
    /// -------
    /// Formula
    ///     Shared immutable native topology.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If clauses or declared variable bounds are invalid.
    /// OverflowError
    ///     If an integer exceeds the native range.
    fn new(clauses: Vec<Vec<i32>>, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or_else(|| {
            clauses
                .iter()
                .flatten()
                .map(|l| l.unsigned_abs() as usize)
                .max()
                .unwrap_or(0)
        });
        Ok(Self {
            inner: Formula::new(n, clauses).map_err(value_error)?,
        })
    }
    #[staticmethod]
    /// Parse DIMACS text with one p cnf N M header and zero-terminated clauses.
    ///
    /// The header counts input clauses before normalization. Comments beginning
    /// with c and blank lines are ignored. Invalid text raises ValueError.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     DIMACS input with one p cnf N M header and zero-terminated clauses.
    ///
    /// Returns
    /// -------
    /// Formula
    ///     Normalized immutable formula retaining declared variable IDs.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If header counts, literal bounds, or clause termination are invalid.
    fn from_dimacs(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Formula::from_dimacs(text).map_err(value_error)?,
        })
    }
    #[staticmethod]
    #[pyo3(signature = (k, alpha, n, seed=1))]
    /// Generate random k-SAT with floor(alpha*n) independent clauses.
    ///
    /// Each clause has k distinct variables sampled uniformly, with independent
    /// unbiased signs. Requires 1 <= k <= n, finite alpha >= 0, and capacity
    /// within packed u32 incidence limits. The unsigned 64-bit seed controls
    /// formula generation, independently of Options.seed. Releases the GIL.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///     Distinct variables per clause, in [1, n].
    /// alpha : float
    ///     Finite nonnegative clause density; m is floor(alpha * n).
    /// n : int
    ///     Declared variable count, at least k and within packed graph limits.
    /// seed : int, default=1
    ///     Unsigned 64-bit formula-generation seed, independent of the solver seed.
    ///
    /// Returns
    /// -------
    /// Formula
    ///     Random k-SAT instance with independently sampled clauses and signs.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If dimensions, density, or packed graph capacity are invalid.
    /// OverflowError
    ///     If integer conversion exceeds the native range.
    fn random(py: Python<'_>, k: usize, alpha: f64, n: usize, seed: u64) -> PyResult<Self> {
        py.allow_threads(|| Formula::random(k, alpha, n, seed))
            .map(|inner| Self { inner })
            .map_err(value_error)
    }
    #[getter]
    /// Declared number of variables, including unused ones.
    ///
    /// Returns
    /// -------
    /// int
    ///     Declared variable count, including unused variables.
    fn n(&self) -> usize {
        self.inner.n()
    }
    #[getter]
    /// Number of clauses after normalization.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of clauses after normalization.
    fn m(&self) -> usize {
        self.inner.m()
    }
    #[getter]
    /// Number of literal incidences after normalization.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of normalized literal incidences.
    fn edges(&self) -> usize {
        self.inner.edges()
    }
    #[getter]
    /// Owned list of normalized clauses, each containing signed one-based literals.
    ///
    /// Returns
    /// -------
    /// list of list of int
    ///     Owned normalized clauses in native incidence order.
    fn clauses(&self) -> Vec<Vec<i32>> {
        self.inner.clauses()
    }
    /// Serialize normalized clauses as DIMACS, preserving the declared n.
    ///
    /// Returns
    /// -------
    /// str
    ///     DIMACS text including the original declared variable count.
    fn to_dimacs(&self) -> String {
        self.inner.to_dimacs()
    }
    /// Return whether a complete 0/1 assignment satisfies every original clause.
    ///
    /// Requires exactly n values. A wrong length, -1 entries, or values other
    /// than 0/1 return False. An empty clause is always unsatisfied.
    ///
    /// Parameters
    /// ----------
    /// assignment : sequence of int
    ///     Complete assignment in variable order, encoded as 0/1.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True exactly when the length is n, values are 0/1, and all clauses
    ///     are satisfied. Other lengths or int8 values return False.
    ///
    /// Raises
    /// ------
    /// OverflowError
    ///     If a value cannot be converted to the native signed 8-bit type.
    /// TypeError
    ///     If the input cannot be converted to a sequence of integers.
    fn is_satisfied(&self, assignment: Vec<i8>) -> bool {
        self.inner.is_satisfied(&assignment)
    }
}

#[pyclass(name = "CancellationToken", module = "pybsp_rust.bsp_rust", frozen)]
/// Thread-safe, one-way cancellation handle obtained from a Solver.
///
/// Call cancel() from another thread during a long native computation. The
/// solver checks cooperatively at sweep/submission and local-search checkpoints.
/// There is no reset method; create another solver for a fresh token.
struct PyCancellationToken {
    inner: CancellationToken,
}
#[pymethods]
impl PyCancellationToken {
    /// Request cancellation. Repeated calls are harmless; the flag stays set.
    ///
    /// Returns
    /// -------
    /// None
    ///     Cancellation stays permanently requested for the associated solver.
    fn cancel(&self) {
        self.inner.cancel();
    }
    #[getter]
    /// Whether cancellation has been requested, even if computation has not stopped.
    ///
    /// Returns
    /// -------
    /// bool
    ///     Whether interruption has been requested; native work may still be stopping.
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

#[pyclass(name = "Solver", module = "pybsp_rust.bsp_rust")]
struct PySolver {
    inner: Solver,
}
#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (formula, options_json=None, execution_json=None))]
    /// Construct native solver state from serialized configuration.
    ///
    /// Parameters
    /// ----------
    /// formula : Formula
    ///     Shared immutable input topology.
    /// options_json : str or None, default=None
    ///     Algorithm settings JSON; None selects Rust defaults.
    /// execution_json : str or None, default=None
    ///     Backend/resource settings JSON; None selects scalar CPU defaults.
    ///
    /// Returns
    /// -------
    /// Solver
    ///     Independent mutable native state.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the formula, settings, or backend setup is invalid.
    fn new(
        py: Python<'_>,
        formula: &PyFormula,
        options_json: Option<&str>,
        execution_json: Option<&str>,
    ) -> PyResult<Self> {
        let options = options(options_json)?;
        let execution = execution(execution_json)?;
        let formula = formula.inner.clone();
        py.allow_threads(|| Solver::with_execution(formula, options, execution))
            .map(|inner| Self { inner })
            .map_err(value_error)
    }
    #[getter]
    /// Return a handle safe to use from another Python thread.
    ///
    /// Returns
    /// -------
    /// CancellationToken
    ///     One-way cooperative cancellation handle.
    fn cancellation_token(&self) -> PyCancellationToken {
        PyCancellationToken {
            inner: self.inner.cancellation_token(),
        }
    }
    #[getter]
    /// Serialize current execution metadata.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON object with backend, precision, resources, counters, and errors.
    fn execution_report(&self) -> PyResult<String> {
        json(self.inner.execution_report())
    }
    #[getter]
    /// Export the current partial assignment.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Owned int8 array of length n with -1/0/1 values.
    fn assignment<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        self.inner.assignment().to_vec().into_pyarray(py)
    }
    #[getter]
    /// Export current clause-to-variable messages.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Owned float64 array in normalized incidence order.
    fn messages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.messages().to_vec().into_pyarray(py)
    }
    #[getter]
    /// Export the current variable surveys.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Owned flat float64 array of length 3*n, in true/false/free order.
    fn surveys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .surveys()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
    #[getter]
    /// Count variables currently unfixed.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of free variables.
    fn free_variables(&self) -> usize {
        self.inner.free_variables()
    }
    #[getter]
    /// Count clauses not satisfied by fixed variables.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of active clauses.
    fn active_clauses(&self) -> usize {
        self.inner.active_clauses()
    }
    /// Update live/frozen messages from a contiguous NumPy array.
    ///
    /// Parameters
    /// ----------
    /// values : numpy.ndarray
    ///     Flat contiguous float64 probabilities of length formula.edges in [0,1].
    ///
    /// Returns
    /// -------
    /// None
    ///     Message state is changed in place.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If length, contiguity, or probabilities are invalid.
    fn set_messages(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner
            .set_messages(values.as_slice().map_err(value_error)?)
            .map_err(value_error)
    }
    /// Fix an explicit zero-based variable decision.
    ///
    /// Parameters
    /// ----------
    /// variable : int
    ///     Currently free variable ID in [0, n).
    /// value : bool
    ///     Desired truth value.
    ///
    /// Returns
    /// -------
    /// None
    ///     Native graph state is simplified; propagation is deferred.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the variable is invalid or already fixed.
    fn fix(&mut self, variable: usize, value: bool) -> PyResult<()> {
        self.inner.fix(variable, value).map_err(value_error)
    }
    /// Release an explicit decision and restore frozen messages.
    ///
    /// Parameters
    /// ----------
    /// variable : int
    ///     Zero-based fixed decision ID; unit-forced variables cannot be released.
    ///
    /// Returns
    /// -------
    /// None
    ///     Native graph state is updated in place.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the variable is invalid, free, or unit-forced.
    fn release(&mut self, variable: usize) -> PyResult<()> {
        self.inner.release(variable).map_err(value_error)
    }
    /// Apply queued unit implications.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of newly forced variables.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     On contradictory residual, interruption, or numerical/device failure.
    fn propagate(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.propagate())
            .map_err(runtime_error)
    }
    /// Run one synchronous message update and refresh snapshots.
    ///
    /// Returns
    /// -------
    /// float
    ///     Largest undamped message change.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     On contradiction, interruption, or numerical/device failure.
    fn sweep(&mut self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.sweep())
            .map_err(runtime_error)
    }
    /// Run a bounded number of raw sweeps with one final snapshot.
    ///
    /// Parameters
    /// ----------
    /// count : int
    ///     Nonnegative sweep count; zero returns 0.0.
    ///
    /// Returns
    /// -------
    /// float
    ///     Final undamped residual.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     On contradiction, interruption, or numerical/device failure.
    /// OverflowError
    ///     If count exceeds the native integer range or is negative.
    fn sweep_many(&mut self, py: Python<'_>, count: usize) -> PyResult<f64> {
        py.allow_threads(|| self.inner.sweep_many(count))
            .map_err(runtime_error)
    }
    /// Attempt an SP fixed point and refresh observables.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON report with sweeps, residual, complexity, and paramagnetic diagnostic.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     On nonconvergence, contradiction, interruption, or numerical/device failure.
    fn converge(&mut self, py: Python<'_>) -> PyResult<String> {
        json(
            &py.allow_threads(|| self.inner.converge())
                .map_err(runtime_error)?,
        )
    }
    /// Run a fixed point and one scheduled BSP action.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON step record with action, counts, residual, and complexity.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the state is terminal or the action fails.
    fn advance(&mut self, py: Python<'_>) -> PyResult<String> {
        json(
            &py.allow_threads(|| self.inner.advance())
                .map_err(runtime_error)?,
        )
    }
    /// Fix a batch using cached survey certainties.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of variables fixed. Call converge first to refresh surveys.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If interruption or a native operational failure prevents the action.
    fn decimate(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.decimate())
    }
    /// Release eligible decisions with low frozen certainty.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of decisions released.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If interruption or a native operational failure prevents the action.
    fn backtrack(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.backtrack())
    }
    /// Continue search until a terminal condition.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON result; only status sat contains a verified complete assignment.
    ///     Operational failures are status records, not UNSAT proofs.
    fn solve(&mut self, py: Python<'_>) -> PyResult<String> {
        json(&py.allow_threads(|| self.inner.solve()))
    }
    /// Extract clauses remaining under the current partial assignment.
    ///
    /// Returns
    /// -------
    /// Formula
    ///     Residual retaining the original variable labels and declared n.
    fn residual_formula(&self, py: Python<'_>) -> PyFormula {
        PyFormula {
            inner: py.allow_threads(|| self.inner.residual_formula()),
        }
    }
}

#[pyfunction]
/// Return native default algorithm settings.
///
/// Returns
/// -------
/// str
///     JSON representation of Rust Options defaults.
fn default_options() -> PyResult<String> {
    json(&Options::default())
}
#[pyfunction]
/// Validate serialized algorithm settings without starting a solver.
///
/// Parameters
/// ----------
/// text : str
///     JSON object accepted by native Options.
///
/// Returns
/// -------
/// None
///     Validation succeeded.
///
/// Raises
/// ------
/// ValueError
///     If JSON or an algorithm setting is invalid.
fn validate_options(text: &str) -> PyResult<()> {
    options(Some(text)).map(|_| ())
}
fn execution(text: Option<&str>) -> PyResult<ExecutionOptions> {
    let value: ExecutionOptions = match text {
        Some(text) => serde_json::from_str(text).map_err(value_error)?,
        None => ExecutionOptions::default(),
    };
    if value.threads > 1024 {
        return Err(value_error("threads must be at most 1024"));
    }
    Ok(value)
}
#[pyfunction]
/// Validate serialized hardware settings.
///
/// Parameters
/// ----------
/// text : str
///     Backend/resource settings JSON.
///
/// Returns
/// -------
/// None
///     Syntax and worker limits are valid; device setup occurs later.
///
/// Raises
/// ------
/// ValueError
///     If JSON, the backend name, or worker limit is invalid.
fn validate_execution(text: &str) -> PyResult<()> {
    execution(Some(text)).map(|_| ())
}
#[pyfunction]
#[pyo3(signature = (formulas, options_json, execution_json=None))]
/// Solve independent native jobs in input order.
///
/// Parameters
/// ----------
/// formulas : sequence of Formula
///     Inputs sharing immutable topology where possible.
/// options_json : sequence of str
///     One validated algorithm-settings JSON object per input.
/// execution_json : str or None, default=None
///     Shared backend and resource settings; profile must be null.
///
/// Returns
/// -------
/// str
///     JSON list of terminal result objects in input order.
///
/// Raises
/// ------
/// ValueError
///     If lengths/settings disagree or backend setup fails.
fn solve_many(
    py: Python<'_>,
    formulas: Vec<Py<PyFormula>>,
    options_json: Vec<String>,
    execution_json: Option<&str>,
) -> PyResult<String> {
    if formulas.len() != options_json.len() {
        return Err(value_error("one Options value is required per formula"));
    }
    let jobs = formulas
        .iter()
        .zip(&options_json)
        .map(|(f, o)| {
            Ok(SolveJob {
                formula: f.borrow(py).inner.clone(),
                options: options(Some(o))?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let execution = execution(execution_json)?;
    json(
        &py.allow_threads(|| crate::solve_many(jobs, execution))
            .map_err(value_error)?,
    )
}
#[pymodule]
fn bsp_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFormula>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyCancellationToken>()?;
    m.add_function(wrap_pyfunction!(default_options, m)?)?;
    m.add_function(wrap_pyfunction!(validate_options, m)?)?;
    m.add_function(wrap_pyfunction!(validate_execution, m)?)?;
    m.add_function(wrap_pyfunction!(solve_many, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
