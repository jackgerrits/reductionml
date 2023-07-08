use pythonize::{depythonize, pythonize};
use std::sync::Arc;

use pyo3::{pyclass, pymethods, types::PyDict, Python};
use reductionml_core::workspace::Configuration;

#[pyclass]
pub(crate) struct Workspace {
    workspace: reductionml_core::workspace::Workspace,
}

#[pymethods]
impl Workspace {
    #[new]
    pub(crate) fn new(args: &PyDict) -> Self {
        let config: Configuration = depythonize(args).unwrap();
        let workspace =
            reductionml_core::workspace::Workspace::create_from_configuration(config).unwrap();
        Self { workspace }
    }
}
