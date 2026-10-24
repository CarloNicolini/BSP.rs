use crate::{CancellationToken, Formula, Options, Solver};
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
struct PyFormula {
    inner: Formula,
}
#[pymethods]
impl PyFormula {
    #[new]
    #[pyo3(signature = (clauses, n=None))]
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
    fn from_dimacs(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Formula::from_dimacs(text).map_err(value_error)?,
        })
    }
    #[staticmethod]
    #[pyo3(signature = (k, alpha, n, seed=1))]
    fn random(py: Python<'_>, k: usize, alpha: f64, n: usize, seed: u64) -> PyResult<Self> {
        py.allow_threads(|| Formula::random(k, alpha, n, seed))
            .map(|inner| Self { inner })
            .map_err(value_error)
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }
    #[getter]
    fn m(&self) -> usize {
        self.inner.m()
    }
    #[getter]
    fn edges(&self) -> usize {
        self.inner.edges()
    }
    #[getter]
    fn clauses(&self) -> Vec<Vec<i32>> {
        self.inner.clauses()
    }
    fn to_dimacs(&self) -> String {
        self.inner.to_dimacs()
    }
    fn is_satisfied(&self, assignment: Vec<i8>) -> bool {
        self.inner.is_satisfied(&assignment)
    }
}

#[pyclass(name = "CancellationToken", module = "pybsp_rust.bsp_rust", frozen)]
struct PyCancellationToken {
    inner: CancellationToken,
}
#[pymethods]
impl PyCancellationToken {
    fn cancel(&self) {
        self.inner.cancel();
    }
    #[getter]
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
    #[pyo3(signature = (formula, options_json=None))]
    fn new(py: Python<'_>, formula: &PyFormula, options_json: Option<&str>) -> PyResult<Self> {
        let options = options(options_json)?;
        let formula = formula.inner.clone();
        py.allow_threads(|| Solver::new(formula, options))
            .map(|inner| Self { inner })
            .map_err(value_error)
    }
    #[getter]
    fn cancellation_token(&self) -> PyCancellationToken {
        PyCancellationToken {
            inner: self.inner.cancellation_token(),
        }
    }
    #[getter]
    fn assignment<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        self.inner.assignment().to_vec().into_pyarray(py)
    }
    #[getter]
    fn messages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.messages().to_vec().into_pyarray(py)
    }
    #[getter]
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
    fn free_variables(&self) -> usize {
        self.inner.free_variables()
    }
    #[getter]
    fn active_clauses(&self) -> usize {
        self.inner.active_clauses()
    }
    fn set_messages(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner
            .set_messages(values.as_slice().map_err(value_error)?)
            .map_err(value_error)
    }
    fn fix(&mut self, variable: usize, value: bool) -> PyResult<()> {
        self.inner.fix(variable, value).map_err(value_error)
    }
    fn release(&mut self, variable: usize) -> PyResult<()> {
        self.inner.release(variable).map_err(value_error)
    }
    fn propagate(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.propagate())
            .map_err(runtime_error)
    }
    fn sweep(&mut self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.sweep())
            .map_err(runtime_error)
    }
    fn converge(&mut self, py: Python<'_>) -> PyResult<String> {
        json(
            &py.allow_threads(|| self.inner.converge())
                .map_err(runtime_error)?,
        )
    }
    fn advance(&mut self, py: Python<'_>) -> PyResult<String> {
        json(
            &py.allow_threads(|| self.inner.advance())
                .map_err(runtime_error)?,
        )
    }
    fn decimate(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.decimate())
    }
    fn backtrack(&mut self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.inner.backtrack())
    }
    fn solve(&mut self, py: Python<'_>) -> PyResult<String> {
        json(&py.allow_threads(|| self.inner.solve()))
    }
    fn residual_formula(&self, py: Python<'_>) -> PyFormula {
        PyFormula {
            inner: py.allow_threads(|| self.inner.residual_formula()),
        }
    }
}

#[pyfunction]
fn default_options() -> PyResult<String> {
    json(&Options::default())
}
#[pyfunction]
fn validate_options(text: &str) -> PyResult<()> {
    options(Some(text)).map(|_| ())
}
#[pymodule]
fn bsp_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFormula>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyCancellationToken>()?;
    m.add_function(wrap_pyfunction!(default_options, m)?)?;
    m.add_function(wrap_pyfunction!(validate_options, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
