use pyo3::prelude::*;
use reductionml_core::PredictionType;

#[pyclass]
#[pyo3(name = "PredictionType")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Type of prediction
///
/// - Scalar - Corresponds with :py:class:`reductionml.ScalarPred`
/// - ActionScores - Corresponds with :py:class:`reductionml.ActionScoresPred`
/// - ActionProbs - Corresponds with :py:class:`reductionml.ActionProbsPred`
/// - Binary - not implemented
pub(crate) enum WrappedPredictionType {
    Scalar,
    Binary,
    ActionScores,
    ActionProbs,
}

impl From<PredictionType> for WrappedPredictionType {
    fn from(x: PredictionType) -> Self {
        match x {
            PredictionType::Scalar => WrappedPredictionType::Scalar,
            PredictionType::Binary => WrappedPredictionType::Binary,
            PredictionType::ActionScores => WrappedPredictionType::ActionScores,
            PredictionType::ActionProbs => WrappedPredictionType::ActionProbs,
        }
    }
}

impl From<WrappedPredictionType> for PredictionType {
    fn from(x: WrappedPredictionType) -> Self {
        match x {
            WrappedPredictionType::Scalar => PredictionType::Scalar,
            WrappedPredictionType::Binary => PredictionType::Binary,
            WrappedPredictionType::ActionScores => PredictionType::ActionScores,
            WrappedPredictionType::ActionProbs => PredictionType::ActionProbs,
        }
    }
}

#[pyclass]
#[pyo3(name = "ScalarPred")]
/// __init__(prediction: float, raw_prediction: float) -> None
///
/// Args:
///     prediction(float): Prediction value (including clamping by the seen range and link function)
///     raw_prediction(float): Raw prediction value
pub(crate) struct WrappedScalarPrediction(reductionml_core::ScalarPrediction);

#[pymethods]
impl WrappedScalarPrediction {
    #[new]
    fn new(prediction: f32, raw_prediction: f32) -> WrappedScalarPrediction {
        WrappedScalarPrediction(reductionml_core::ScalarPrediction {
            prediction,
            raw_prediction,
        })
    }

    #[getter]
    fn get_prediction(&self) -> f32 {
        self.0.prediction
    }

    #[getter]
    fn get_raw_prediction(&self) -> f32 {
        self.0.raw_prediction
    }

    fn __str__(&self) -> String {
        format!("{}, {}", self.0.prediction, self.0.raw_prediction)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScalarPred(prediction={}, raw_prediction={})",
            self.0.prediction, self.0.raw_prediction
        )
    }
}

#[pyclass]
#[pyo3(name = "ActionScoresPred")]
/// __init__(value: List[Tuple[int, float]]) -> None
///
/// Args:
///     value: A list of tuples of the form (action, score)
pub(crate) struct WrappedActionScoresPrediction(reductionml_core::ActionScoresPrediction);

#[pymethods]
impl WrappedActionScoresPrediction {
    #[new]
    fn new(value: Vec<(usize, f32)>) -> WrappedActionScoresPrediction {
        WrappedActionScoresPrediction(reductionml_core::ActionScoresPrediction(value))
    }

    #[getter]
    fn get_value(&self) -> Vec<(usize, f32)> {
        self.0 .0.clone()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0 .0)
    }

    fn __repr__(&self) -> String {
        format!("ActionScoresPred(value={:?})", self.0 .0)
    }
}

#[pyclass]
#[pyo3(name = "ActionProbsPred")]
/// __init__(value: List[Tuple[int, float]]) -> None
///
/// Args:
///     value: A list of tuples of the form (action, probability)
pub(crate) struct WrappedActionProbsPrediction(reductionml_core::ActionProbsPrediction);

#[pymethods]
impl WrappedActionProbsPrediction {
    #[new]
    fn new(value: Vec<(usize, f32)>) -> WrappedActionProbsPrediction {
        WrappedActionProbsPrediction(reductionml_core::ActionProbsPrediction(value))
    }

    #[getter]
    fn get_value(&self) -> Vec<(usize, f32)> {
        self.0 .0.clone()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0 .0)
    }

    fn __repr__(&self) -> String {
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
