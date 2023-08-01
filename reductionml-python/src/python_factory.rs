use core::num;
use std::sync::Arc;

use pyo3::prelude::*;

use pyo3::types::PyDict;
use pyo3::{PyAny, Python};
use pythonize::{depythonize, pythonize};
use reductionml_core::error::Result;
use reductionml_core::global_config::GlobalConfig;
use reductionml_core::reduction::ReductionWrapper;
use reductionml_core::reduction_factory::{PascalCaseString, ReductionConfig, ReductionFactory};
use reductionml_core::ModelIndex;
use serde_json::json;

use crate::workspace::WrappedReductionTypesDescription;

struct PythonReductionFactory {
    create_func: PyAny,
    typename: PascalCaseString,
}

// #[derive(Deserialize, DefaultFromSerde, Serialize, Debug, Clone, JsonSchema, Builder)]
// #[serde(deny_unknown_fields)]
// #[serde(rename_all = "camelCase")]
#[derive(Debug)]
pub struct PyReductionConfig {
    py_dict: serde_json::Value,
}

impl ReductionConfig for PyReductionConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn typename(&self) -> PascalCaseString {
        "Unknown".to_owned().try_into().unwrap()
    }
}

impl ReductionFactory for PythonReductionFactory {
    fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>> {
        Ok(Box::new(PyReductionConfig {
            py_dict: value.clone(),
        }))
    }

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config: &PyReductionConfig =
            config.as_any().downcast_ref::<PyReductionConfig>().unwrap();

        Python::with_gil(|py| {
            let dict = pythonize(py, &config.py_dict).unwrap();
            let dict: Py<PyDict> = dict.extract(py).unwrap();
            let num_models: u8 = num_models_above.into();
            let glb_config: Py<PyAny> = pythonize(py, global_config).unwrap();
            let args = (dict, glb_config, num_models);

            let (return_value, num2): (PyObject, PyObject) =
                self.create_func.call1(args).unwrap().extract().unwrap();
            let re: Py<WrappedReductionTypesDescription> = num2.extract(py).unwrap();
            let a: WrappedReductionTypesDescription = re.try_borrow(py).unwrap().clone();

            todo!()
        })
    }

    fn typename(&self) -> PascalCaseString {
        self.typename.clone()
    }

    fn get_config_default(&self) -> serde_json::Value {
        // TODO expose python default
        json!({})
    }
}

struct PythonReduction {
    obj: Py<PyAny>,
}
