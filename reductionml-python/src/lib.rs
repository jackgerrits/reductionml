use pyo3::prelude::*;
use reductionml_core::{object_pool::Pool, FeaturesType};

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

#[pyclass]
#[pyo3(name = "FeaturesType")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Type of features
///
/// - SparseSimple - Corresponds with :py:class:`reductionml.SparseFeatures`
/// - SparseCbAdf - Corresponds with :py:class:`reductionml.CbAdfFeatures`
pub(crate) enum WrappedFeaturesType {
    SparseSimple,
    SparseCbAdf,
}

impl From<FeaturesType> for WrappedFeaturesType {
    fn from(x: FeaturesType) -> Self {
        match x {
            FeaturesType::SparseSimple => WrappedFeaturesType::SparseSimple,
            FeaturesType::SparseCBAdf => WrappedFeaturesType::SparseCbAdf,
        }
    }
}

impl From<WrappedFeaturesType> for FeaturesType {
    fn from(x: WrappedFeaturesType) -> Self {
        match x {
            WrappedFeaturesType::SparseSimple => FeaturesType::SparseSimple,
            WrappedFeaturesType::SparseCbAdf => FeaturesType::SparseCBAdf,
        }
    }
}

// expose version
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

#[pymodule]
fn _reductionml(_py: Python, m: &PyModule) -> PyResult<()> {
    // Workspace
    m.add_class::<workspace::WrappedWorkspace>()?;
    m.add_class::<workspace::WrappedReductionTypesDescription>()?;

    // Features
    m.add_class::<WrappedFeaturesType>()?;
    m.add_class::<features::WrappedSparseFeatures>()?;
    m.add_class::<features::WrappedCbAdfFeatures>()?;

    // Labels
    m.add_class::<labels::WrappedSimpleLabel>()?;
    m.add_class::<labels::WrappedCBLabel>()?;
    m.add_class::<labels::WrappedLabelType>()?;

    // Predictions
    m.add_class::<predictions::WrappedScalarPrediction>()?;
    m.add_class::<predictions::WrappedActionProbsPrediction>()?;
    m.add_class::<predictions::WrappedActionScoresPrediction>()?;
    m.add_class::<predictions::WrappedPredictionType>()?;

    // Parsers
    m.add_class::<parsers::FormatType>()?;
    m.add_class::<parsers::WrappedParserTextOnly>()?;
    m.add_class::<parsers::WrappedParserTextAndJson>()?;
    m.add_function(wrap_pyfunction!(parsers::create_parser, m)?)?;

    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
