use std::any::Any;

use schemars::{JsonSchema, schema::{RootSchema, Schema, SchemaObject}, gen::SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    error::{Error, Result},
    global_config::GlobalConfig,
    reduction::ReductionWrapper,
    reduction_registry::REDUCTION_REGISTRY,
    ModelIndex,
};

// This intentionally does not derive JsonSchema
// Use gen_json_reduction_config_schema instead with schema_with
#[derive(Serialize, Deserialize)]
pub struct JsonReductionConfig {
    typename: String,
    config: serde_json::Value,
}

impl JsonReductionConfig {
    pub fn new(typename: String, config: serde_json::Value) -> Self {
        JsonReductionConfig { typename, config }
    }
}

// impl TryFrom<serde_json::Value> for JsonReductionConfig {
//     type Error = Error;

//     fn try_from(value: serde_json::Value) -> std::result::Result<Self, Self::Error> {
//         let typename = value["typename"]
//             .as_str()
//             .ok_or(Error::InvalidArgument(
//                 "typename must be a string".to_owned(),
//             ))?
//             .to_string();
//         let config = value["config"].clone();
//         Ok(JsonReductionConfig { typename, config })
//     }
// }

// impl Into<serde_json::Value> for JsonReductionConfig {
//     fn into(self) -> serde_json::Value {
//         json!({
//             "typename": self.typename,
//             "config": self.config
//         })
//     }
// }

impl JsonReductionConfig {
    pub fn typename(&self) -> String {
        self.typename.to_owned()
    }
    pub fn json_value(&self) -> &serde_json::Value {
        &self.config
    }
}

pub trait ReductionConfig: Any {
    fn as_any(&self) -> &dyn Any;
    fn typename(&self) -> String;
}

pub trait ReductionFactory {
    fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>>;
    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper>;
    fn typename(&self) -> String;
    fn get_config_schema(&self) -> RootSchema;
}

#[macro_export]
macro_rules! impl_default_factory_functions {
    ($typename: expr, $config_type: ident) => {
        fn typename(&self) -> String {
            $typename.to_owned()
        }
        fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>> {
            let res: $config_type = serde_json::from_value(value.clone()).unwrap();
            Ok(Box::new(res))
        }

        fn get_config_schema(&self) -> RootSchema
        {
            schema_for!($config_type)
        }
    };
}

pub fn parse_config(config: &JsonReductionConfig) -> Result<Box<dyn ReductionConfig>> {
    match REDUCTION_REGISTRY.read().unwrap().get(&config.typename) {
        Some(factory) => factory.parse_config(&config.json_value()),
        None => Err(crate::error::Error::InvalidArgument(format!(
            "Unknown reduction type: {}",
            &config.typename
        ))),
    }
}

pub fn create_reduction(
    config: &dyn ReductionConfig,
    global_config: &GlobalConfig,
    num_models_above: ModelIndex,
) -> Result<ReductionWrapper> {
    match REDUCTION_REGISTRY.read().unwrap().get(&config.typename()) {
        Some(factory) => factory.create(config, global_config, num_models_above),
        None => Err(crate::error::Error::InvalidArgument(format!(
            "Unknown reduction type: {}",
            config.typename()
        ))),
    }
}