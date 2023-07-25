use pyo3::prelude::*;
use pythonize::pythonize;
use reductionml_core::reduction_registry::REDUCTION_REGISTRY;
use serde_json::json;

fn get_type(prop: &serde_json::Value) -> String {
    match prop {
        serde_json::Value::Bool(_) => "bool".to_string(),
        serde_json::Value::Number(_) => "number".to_string(),
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "array".to_string(),
        serde_json::Value::Object(_) => "reduction".to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn get_default_value(prop: &serde_json::Value) -> serde_json::Value {
    match prop {
        serde_json::Value::Bool(value) => value.to_string().into(),
        // TODO: fix this for integers.
        serde_json::Value::Number(value) => ((value.as_f64().unwrap() * 100.0).round() / 100.0)
            .to_string()
            .into(),
        serde_json::Value::String(value) => value.to_string().into(),
        serde_json::Value::Array(_) => todo!(),
        // For now we will assume this always corresponds to a reduction
        serde_json::Value::Object(obj) => match obj.get("typename") {
            Some(name) => name.as_str().unwrap().to_owned().into(),
            None => prop.clone(),
        },
        serde_json::Value::Null => "null".to_string().into(),
    }
}

#[pyfunction]
fn get_reduction_info(name: &str) -> PyResult<PyObject> {
    let mut default_config = REDUCTION_REGISTRY.lock().get(name).unwrap().get_config_default();

    let mut props = vec![];
    for (key, prop) in default_config.as_object().unwrap() {
        props.push(json!({
            "name": key,
            "type": get_type(prop),
            "default": get_default_value(prop)
        }));
    }

    Python::with_gil(|py| {
        let data = pythonize(py, &props).unwrap();
        Ok(data.into_py(py))
    })
}

#[pymodule]
fn reductionml_docs_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_reduction_info, m)?)?;
    Ok(())
}
