use pyo3::prelude::*;

use pythonize::{depythonize, pythonize};

use pyo3::{pyclass, pymethods, types::PyDict, IntoPy, Python};
use reductionml_core::{
    reduction::{ReductionTypeDescription, ReductionTypeDescriptionBuilder},
    workspace::Configuration,
    Label,
};

use crate::{
    features::WrappedFeatures,
    labels::{WrappedLabel, WrappedLabelType},
    parsers::{create_parser, FormatType, WrappedParser},
    predictions::{WrappedPrediction, WrappedPredictionType},
    WrappedFeaturesType,
};

#[pyclass]
#[pyo3(name = "ReductionTypesDescription")]
#[derive(Clone)]
/// __init__(self, input_label_type: LabelType, output_prediction_type: PredictionType, input_features_type: FeaturesType, output_label_type: Optional[LabelType] = None, input_prediction_type: Optional[PredictionType] = None, output_features_type: Optional[FeaturesType] = None)
pub(crate) struct WrappedReductionTypesDescription(ReductionTypeDescription);

impl Into<ReductionTypeDescription> for WrappedReductionTypesDescription {
    fn into(self) -> ReductionTypeDescription {
        self.0
    }
}

#[pymethods]
impl WrappedReductionTypesDescription {
    #[new]
    pub(crate) fn new(
        input_label_type: WrappedLabelType,
        output_prediction_type: WrappedPredictionType,
        input_features_type: WrappedFeaturesType,
        output_label_type: Option<WrappedLabelType>,
        input_prediction_type: Option<WrappedPredictionType>,
        output_features_type: Option<WrappedFeaturesType>,
    ) -> Self {
        let builder = ReductionTypeDescriptionBuilder::new(
            input_label_type.into(),
            input_features_type.into(),
            output_prediction_type.into(),
        );
        let builder = if let Some(output_label_type) = output_label_type {
            builder.with_output_label_type(output_label_type.into())
        } else {
            builder
        };
        let builder = if let Some(input_prediction_type) = input_prediction_type {
            builder.with_input_prediction_type(input_prediction_type.into())
        } else {
            builder
        };
        let builder = if let Some(output_features_type) = output_features_type {
            builder.with_output_features_type(output_features_type.into())
        } else {
            builder
        };
        Self(builder.build())
    }

    #[getter]
    /// The label type expected as input for this reduction
    ///
    /// Returns:
    ///    LabelType:
    pub(crate) fn input_label_type(&self) -> WrappedLabelType {
        self.0.input_label_type().into()
    }

    #[getter]
    /// The label type expected as output for this reduction. If this is a base reduction, this will be None
    ///
    /// Returns:
    ///   Optional[LabelType]:
    pub(crate) fn output_label_type(&self) -> Option<WrappedLabelType> {
        self.0.output_label_type().map(Into::into)
    }

    #[getter]
    /// The prediction type expected as input for this reduction. If this is a base reduction, this will be None
    ///
    /// Returns:
    ///   Optional[PredictionType]:
    pub(crate) fn input_prediction_type(&self) -> Option<WrappedPredictionType> {
        self.0.input_prediction_type().map(Into::into)
    }

    #[getter]
    /// The prediction type expected as output for this reduction
    ///
    /// Returns:
    ///   PredictionType:
    pub(crate) fn output_prediction_type(&self) -> WrappedPredictionType {
        self.0.output_prediction_type().into()
    }

    #[getter]
    /// The features type expected as input for this reduction
    ///
    /// Returns:
    ///   FeaturesType:
    pub(crate) fn input_features_type(&self) -> WrappedFeaturesType {
        self.0.input_features_type().into()
    }

    #[getter]
    /// The features type expected as output for this reduction. If this is a base reduction, this will be None
    ///
    /// Returns:
    ///   Optional[FeaturesType]:
    pub(crate) fn output_features_type(&self) -> Option<WrappedFeaturesType> {
        self.0.output_features_type().map(Into::into)
    }
}

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
            features_type.into(),
            label_type.into(),
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

    pub(crate) fn get_entry_reduction_types(&self) -> WrappedReductionTypesDescription {
        WrappedReductionTypesDescription(self.0.get_entry_reduction().types().clone())
    }
}
