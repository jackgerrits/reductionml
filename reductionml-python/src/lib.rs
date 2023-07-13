use pyo3::prelude::*;
use reductionml_core::object_pool::Pool;

use std::sync::Arc;

use once_cell::sync::Lazy;

pub(crate) mod features;
pub(crate) mod labels;
pub(crate) mod parsers;
pub(crate) mod predictions;
pub(crate) mod workspace;

pub static SPARSE_FEATURES_POOL: Lazy<
    Arc<Pool<reductionml_core::sparse_namespaced_features::SparseFeatures>>,
> = Lazy::new(|| Arc::new(Pool::new()));

pub(crate) struct WrappedError(reductionml_core::error::Error);

impl From<reductionml_core::error::Error> for WrappedError {
    fn from(err: reductionml_core::error::Error) -> WrappedError {
        WrappedError(err)
    }
}

impl From<WrappedError> for PyErr {
    fn from(err: WrappedError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err.0))
    }
}

#[pymodule]
fn _reductionml(_py: Python, m: &PyModule) -> PyResult<()> {
    // Workspace
    m.add_class::<workspace::WrappedWorkspace>()?;

    // Features
    m.add_class::<features::WrappedSparseFeatures>()?;
    m.add_class::<features::WrappedCbAdfFeatures>()?;

    // Labels
    m.add_class::<labels::WrappedSimpleLabel>()?;
    m.add_class::<labels::WrappedCBLabel>()?;

    // Predictions
    m.add_class::<predictions::WrappedScalarPrediction>()?;
    m.add_class::<predictions::WrappedActionProbsPrediction>()?;
    m.add_class::<predictions::WrappedActionScoresPrediction>()?;

    // Parsers
    m.add_class::<parsers::FormatType>()?;
    m.add_class::<parsers::ReductionType>()?;
    m.add_class::<parsers::WrappedParserTextOnly>()?;
    m.add_class::<parsers::WrappedParserTextAndJson>()?;
    m.add_function(wrap_pyfunction!(parsers::create_parser, m)?)?;

    Ok(())
}
