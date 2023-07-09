use pyo3::prelude::*;
use reductionml_core::{
    error::Error, object_pool::PoolReturnable, sparse_namespaced_features::SparseFeatures,
    CBAdfFeatures, Features,
};

use crate::SPARSE_FEATURES_POOL;

#[pyclass]
#[pyo3(name = "SparseFeatures")]
pub(crate) struct WrappedSparseFeatures(
    Option<reductionml_core::sparse_namespaced_features::SparseFeatures>,
);

#[pyclass]
#[pyo3(name = "CbAdfFeatures")]
pub(crate) struct WrappedCbAdfFeatures(Option<reductionml_core::CBAdfFeatures>);

#[pymethods]
impl WrappedSparseFeatures {
    #[new]
    pub(crate) fn new() -> Self {
        Self(Some(SPARSE_FEATURES_POOL.as_ref().get_object()))
    }
}

impl Drop for WrappedSparseFeatures {
    fn drop(&mut self) {
        if let Some(features) = self.0.take() {
            features.clear_and_return_object(SPARSE_FEATURES_POOL.as_ref());
        }
    }
}

#[pymethods]
impl WrappedCbAdfFeatures {
    #[new]
    pub(crate) fn new() -> Self {
        Self(Some(reductionml_core::CBAdfFeatures::default()))
    }
}

impl Drop for WrappedCbAdfFeatures {
    fn drop(&mut self) {
        if let Some(features) = self.0.take() {
            features.clear_and_return_object(SPARSE_FEATURES_POOL.as_ref());
        }
    }
}

#[derive(FromPyObject)]
pub(crate) enum WrappedFeatures<'a> {
    SparseSimpleRef(PyRefMut<'a, WrappedSparseFeatures>),
    CbAdfFeaturesRef(PyRefMut<'a, WrappedCbAdfFeatures>),
}

impl WrappedFeatures<'_> {
    // Not really named quite right, but I wasn't sure how to express this.
    pub(crate) fn to_features(&mut self) -> Features {
        match self {
            WrappedFeatures::SparseSimpleRef(ref mut r) => {
                Features::SparseSimpleRef(r.0.as_mut().unwrap())
            }
            WrappedFeatures::CbAdfFeaturesRef(ref mut r) => {
                Features::SparseCBAdfRef(r.0.as_mut().unwrap())
            }
        }
    }
}

pub(crate) enum WrappedFeaturesForReturn {
    SparseSimple(SparseFeatures),
    CbAdfFeatures(CBAdfFeatures),
}

impl TryInto<WrappedFeaturesForReturn> for Features<'_> {
    type Error = Error;

    fn try_into(self) -> Result<WrappedFeaturesForReturn, Self::Error> {
        match self {
            Features::SparseSimple(feats) => Ok(WrappedFeaturesForReturn::SparseSimple(feats)),
            Features::SparseSimpleRef(_) => todo!(),
            Features::SparseCBAdf(feats) => Ok(WrappedFeaturesForReturn::CbAdfFeatures(feats)),
            Features::SparseCBAdfRef(_) => todo!(),
        }
    }
}

impl IntoPy<PyObject> for WrappedFeaturesForReturn {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            WrappedFeaturesForReturn::SparseSimple(feats) => {
                WrappedSparseFeatures(Some(feats)).into_py(py)
            }
            WrappedFeaturesForReturn::CbAdfFeatures(feats) => {
                WrappedCbAdfFeatures(Some(feats)).into_py(py)
            }
        }
    }
}
