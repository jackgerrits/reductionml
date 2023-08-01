use pyo3::prelude::*;
use reductionml_core::{Label, LabelType};

#[pyclass]
#[pyo3(name = "LabelType")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Type of label
///
/// - Simple - Corresponds with :py:class:`reductionml.SimpleLabel`
/// - CB - Corresponds with :py:class:`reductionml.CbLabel`
/// - Binary - not implemented
pub(crate) enum WrappedLabelType {
    Simple,
    Binary,
    CB,
}

impl From<LabelType> for WrappedLabelType {
    fn from(x: LabelType) -> Self {
        match x {
            LabelType::Simple => WrappedLabelType::Simple,
            LabelType::Binary => WrappedLabelType::Binary,
            LabelType::CB => WrappedLabelType::CB,
        }
    }
}

impl From<WrappedLabelType> for LabelType {
    fn from(x: WrappedLabelType) -> Self {
        match x {
            WrappedLabelType::Simple => LabelType::Simple,
            WrappedLabelType::Binary => LabelType::Binary,
            WrappedLabelType::CB => LabelType::CB,
        }
    }
}

#[pyclass]
#[derive(Clone)]
#[pyo3(name = "SimpleLabel")]
/// __init__(value: float, weight: float = 1.0) -> None
///
/// Args:
///     value(float): Label value
///     weight(float): Label weight
///
pub(crate) struct WrappedSimpleLabel(reductionml_core::SimpleLabel);

impl Into<Label> for WrappedSimpleLabel {
    fn into(self) -> Label {
        Label::Simple(self.0)
    }
}

#[pymethods]
impl WrappedSimpleLabel {
    #[new]
    #[pyo3(signature = (value, weight = 1.0))]
    pub(crate) fn new(value: f32, weight: f32) -> Self {
        Self(reductionml_core::SimpleLabel::new(value, weight))
    }

    #[getter]
    /// The label's value
    ///
    /// Returns:
    ///     float:
    fn get_value(&self) -> f32 {
        self.0.value()
    }

    #[getter]
    /// Weight of example to be used in update
    ///
    /// Returns:
    ///     float:
    fn get_weight(&self) -> f32 {
        self.0.weight()
    }

    fn __str__(&self) -> String {
        format!("{}, {}", self.0.value(), self.0.weight())
    }

    fn __repr__(&self) -> String {
        format!(
            "SimpleLabel(value={}, weight={})",
            self.0.value(),
            self.0.weight()
        )
    }
}

#[pyclass]
#[pyo3(name = "CbLabel")]
#[derive(Clone)]
/// __init__(action: int, cost: float, probability: float) -> None
///
/// Args:
///     action(int): Chosen action (zero based)
///     cost(float): Cost of chosen action
///     probability(float): Probability of chosen action
///
pub(crate) struct WrappedCBLabel(reductionml_core::CBLabel);

impl Into<Label> for WrappedCBLabel {
    fn into(self) -> Label {
        Label::CB(self.0)
    }
}

#[pymethods]
impl WrappedCBLabel {
    #[new]
    pub(crate) fn new(action: usize, cost: f32, probability: f32) -> Self {
        Self(reductionml_core::CBLabel::new(action, cost, probability))
    }

    #[getter]
    /// The label's action
    ///
    /// Returns:
    ///     int:
    fn get_action(&self) -> usize {
        self.0.action()
    }

    #[getter]
    /// The label's cost
    ///
    /// Returns:
    ///    float:
    fn get_cost(&self) -> f32 {
        self.0.cost()
    }

    #[getter]
    /// The label's probability
    ///
    /// Returns:
    ///    float:
    fn get_probability(&self) -> f32 {
        self.0.probability()
    }

    fn __str__(&self) -> String {
        format!(
            "{}, {}, {}",
            self.0.action(),
            self.0.cost(),
            self.0.probability()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "CbLabel(action={}, cost={}, probability={})",
            self.0.action(),
            self.0.cost(),
            self.0.probability()
        )
    }
}

#[derive(FromPyObject)]
pub(crate) enum WrappedLabel {
    Simple(WrappedSimpleLabel),
    CB(WrappedCBLabel),
}

impl From<Label> for WrappedLabel {
    fn from(label: Label) -> Self {
        match label {
            Label::Simple(lbl) => WrappedLabel::Simple(WrappedSimpleLabel(lbl)),
            Label::CB(lbl) => WrappedLabel::CB(WrappedCBLabel(lbl)),
            _ => todo!(),
        }
    }
}

impl From<WrappedLabel> for Label {
    fn from(label: WrappedLabel) -> Self {
        match label {
            WrappedLabel::Simple(lbl) => Label::Simple(lbl.0),
            WrappedLabel::CB(lbl) => Label::CB(lbl.0),
        }
    }
}

impl IntoPy<PyObject> for WrappedLabel {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            WrappedLabel::Simple(lbl) => lbl.into_py(py),
            WrappedLabel::CB(lbl) => lbl.into_py(py),
        }
    }
}
