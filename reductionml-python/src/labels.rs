use pyo3::prelude::*;
use reductionml_core::Label;

#[pyclass]
#[derive(Clone)]
#[pyo3(name = "SimpleLabel")]
pub(crate) struct WrappedSimpleLabel(reductionml_core::SimpleLabel);

impl Into<Label> for WrappedSimpleLabel {
    fn into(self) -> Label {
        Label::Simple(self.0)
    }
}

#[pymethods]
impl WrappedSimpleLabel {
    #[new]
    pub(crate) fn new(value: f32, weight: f32) -> Self {
        Self(reductionml_core::SimpleLabel::new(value, weight))
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.0.value()
    }

    #[getter]
    fn get_weight(&self) -> f32 {
        self.0.weight()
    }
}

#[pyclass]
#[pyo3(name = "CbLabel")]
#[derive(Clone)]
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
    fn get_action(&self) -> usize {
        self.0.action()
    }

    #[getter]
    fn get_cost(&self) -> f32 {
        self.0.cost()
    }

    #[getter]
    fn get_probability(&self) -> f32 {
        self.0.probability()
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
