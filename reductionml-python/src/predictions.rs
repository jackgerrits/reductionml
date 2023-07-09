use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "ScalarPred")]
pub(crate) struct WrappedScalarPrediction(reductionml_core::ScalarPrediction);

#[pymethods]
impl WrappedScalarPrediction {
    #[getter]
    fn get_prediction(&self) -> f32 {
        self.0.prediction
    }

    #[getter]
    fn get_raw_prediction(&self) -> f32 {
        self.0.raw_prediction
    }

    fn __str__(&self) -> String {
        format!(
            "ScalarPred(prediction={}, raw_prediction={})",
            self.0.prediction, self.0.raw_prediction
        )
    }
}

#[pyclass]
#[pyo3(name = "ActionScoresPred")]
pub(crate) struct WrappedActionScoresPrediction(reductionml_core::ActionScoresPrediction);

#[pymethods]
impl WrappedActionScoresPrediction {
    #[getter]
    fn get_value(&self) -> Vec<(usize, f32)> {
        self.0 .0.clone()
    }

    fn __str__(&self) -> String {
        format!("ActionScoresPred(value={:?})", self.0 .0)
    }
}

#[pyclass]
#[pyo3(name = "ActionProbsPred")]
pub(crate) struct WrappedActionProbsPrediction(reductionml_core::ActionProbsPrediction);

#[pymethods]
impl WrappedActionProbsPrediction {
    #[getter]
    fn get_value(&self) -> Vec<(usize, f32)> {
        self.0 .0.clone()
    }

    fn __str__(&self) -> String {
        format!("ActionProbsPred(value={:?})", self.0 .0)
    }
}

pub(crate) struct WrappedPrediction(reductionml_core::Prediction);

impl Into<WrappedPrediction> for reductionml_core::Prediction {
    fn into(self) -> WrappedPrediction {
        WrappedPrediction(self)
    }
}

impl IntoPy<PyObject> for WrappedPrediction {
    fn into_py(self, py: Python) -> PyObject {
        match self.0 {
            reductionml_core::Prediction::Scalar(pred) => WrappedScalarPrediction(pred).into_py(py),
            reductionml_core::Prediction::Binary(_) => todo!(),
            reductionml_core::Prediction::ActionScores(pred) => {
                WrappedActionScoresPrediction(pred).into_py(py)
            }
            reductionml_core::Prediction::ActionProbs(pred) => {
                WrappedActionProbsPrediction(pred).into_py(py)
            }
        }
    }
}
