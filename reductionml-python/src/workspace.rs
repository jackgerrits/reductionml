use pyo3::prelude::*;

use pythonize::{depythonize, pythonize};

use pyo3::{pyclass, pymethods, types::PyDict, IntoPy, Python};
use reductionml_core::{workspace::Configuration, Label};

use crate::{
    features::WrappedFeatures,
    labels::WrappedLabel,
    parsers::{create_parser, FormatType, WrappedParser},
    predictions::WrappedPrediction,
};

#[pyclass]
#[pyo3(name = "Workspace")]
pub(crate) struct WrappedWorkspace(pub(crate) reductionml_core::workspace::Workspace);

#[pymethods]
impl WrappedWorkspace {
    #[staticmethod]
    pub(crate) fn create_from_config(config: &PyDict) -> Self {
        let config: Configuration = depythonize(config).unwrap();
        let workspace = reductionml_core::workspace::Workspace::new(config).unwrap();
        Self(workspace)
    }

    #[staticmethod]
    pub(crate) fn create_from_model(data: Vec<u8>) -> Self {
        let workspace = reductionml_core::workspace::Workspace::create_from_model(&data).unwrap();
        Self(workspace)
    }

    #[staticmethod]
    pub(crate) fn create_from_json_model(model_json: &PyDict) -> Self {
        let data = depythonize(model_json).unwrap();
        let workspace =
            reductionml_core::workspace::Workspace::deserialize_from_json(&data).unwrap();
        Self(workspace)
    }

    pub(crate) fn serialize(&self) -> PyResult<Vec<u8>> {
        let data = self.0.serialize_model().unwrap();
        Ok(data)
    }

    pub(crate) fn serialize_to_json(&self) -> PyResult<PyObject> {
        let data = self.0.serialize_to_json().unwrap();
        Python::with_gil(|py| {
            let data = pythonize(py, &data).unwrap();
            Ok(data.into_py(py))
        })
    }

    pub(crate) fn create_parser(&self, format_type: FormatType) -> Result<WrappedParser, PyErr> {
        let features_type = self.0.get_entry_reduction().types().input_features_type();
        let label_type = self.0.get_entry_reduction().types().input_label_type();
        let hash_seed = self.0.global_config().hash_seed();
        let num_bits = self.0.global_config().num_bits();

        create_parser(
            format_type,
            (features_type, label_type).try_into()?,
            hash_seed,
            num_bits,
        )
    }

    pub(crate) fn predict(&self, mut features: WrappedFeatures) -> PyResult<WrappedPrediction> {
        let mut feats = features.to_features();
        let pred = self.0.predict(&mut feats);
        Ok(pred.into())
    }

    pub(crate) fn predict_then_learn(
        &mut self,
        mut features: WrappedFeatures,
        label: WrappedLabel,
    ) -> PyResult<WrappedPrediction> {
        let mut feats = features.to_features();
        let label: Label = label.into();
        let pred = self.0.predict_then_learn(&mut feats, &label);
        Ok(pred.into())
    }

    pub(crate) fn learn(
        &mut self,
        mut features: WrappedFeatures,
        label: WrappedLabel,
    ) -> PyResult<()> {
        let mut feats = features.to_features();
        let label: Label = label.into();
        self.0.learn(&mut feats, &label);
        Ok(())
    }
}
