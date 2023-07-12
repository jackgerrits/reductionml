use std::{
    any::Any,
    fmt::{Display, Formatter},
};

use schemars::schema::RootSchema;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    error::{Error, Result},
    global_config::GlobalConfig,
    reduction::ReductionWrapper,
    reduction_registry::REDUCTION_REGISTRY,
    ModelIndex,
};

// This intentionally does not derive JsonSchema
// Use gen_json_reduction_config_schema instead with schema_with
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonReductionConfig {
    typename: PascalCaseString,
    config: serde_json::Value,
}

impl<'de> Deserialize<'de> for JsonReductionConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut json: serde_json::value::Value =
            serde_json::value::Value::deserialize(deserializer)?;
        let typename = json
            .get("typename")
            .expect("type")
            .as_str()
            .unwrap()
            .to_string()
            .try_into()
            .map_err(|x| serde::de::Error::custom(x))?;
        let config = json.get_mut("value");

        match config {
            Some(config) => Ok(JsonReductionConfig::new(typename, config.take())),
            None => {
                let default_config = REDUCTION_REGISTRY
                    .read()
                    .as_ref()
                    .unwrap()
                    .get(&typename.0)
                    .ok_or_else(|| {
                        serde::de::Error::custom(format!("Unknown reduction type: {}", typename))
                    })?
                    .get_config_default();
                Ok(JsonReductionConfig::new(typename, default_config))
            }
        }
    }
}

impl JsonReductionConfig {
    pub fn new(typename: PascalCaseString, config: serde_json::Value) -> Self {
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
        self.typename.as_ref().to_owned()
    }
    pub fn json_value(&self) -> &serde_json::Value {
        &self.config
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PascalCaseString(String);

impl TryFrom<String> for PascalCaseString {
    type Error = Error;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(Error::InvalidArgument(
                "typename must not be empty".to_owned(),
            ));
        }
        if !value.chars().next().unwrap().is_ascii_uppercase() {
            return Err(Error::InvalidArgument(
                "typename must start with an uppercase letter".to_owned(),
            ));
        }
        if value.chars().any(|c| !c.is_ascii_alphanumeric()) {
            return Err(Error::InvalidArgument(
                "typename must only contain alphanumeric characters".to_owned(),
            ));
        }
        Ok(PascalCaseString(value))
    }
}

impl TryFrom<&str> for PascalCaseString {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(Error::InvalidArgument(
                "typename must not be empty".to_owned(),
            ));
        }
        if !value.chars().next().unwrap().is_ascii_uppercase() {
            return Err(Error::InvalidArgument(
                "typename must start with an uppercase letter".to_owned(),
            ));
        }
        if value.chars().any(|c| !c.is_ascii_alphanumeric()) {
            return Err(Error::InvalidArgument(
                "typename must only contain alphanumeric characters".to_owned(),
            ));
        }
        Ok(PascalCaseString(value.to_owned()))
    }
}

impl From<PascalCaseString> for String {
    fn from(value: PascalCaseString) -> Self {
        value.0
    }
}

impl AsRef<str> for PascalCaseString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Display for PascalCaseString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.clone())
    }
}

pub trait ReductionConfig: Any {
    fn as_any(&self) -> &dyn Any;
    fn typename(&self) -> PascalCaseString;
}

pub trait ReductionFactory {
    fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>>;
    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper>;
    fn typename(&self) -> PascalCaseString;
    fn get_config_schema(&self) -> RootSchema;
    fn get_config_default(&self) -> serde_json::Value;
    fn get_suggested_metrics(&self) -> Vec<String> {
        vec![]
    }
}

#[macro_export]
macro_rules! impl_default_factory_functions {
    ($typename: expr, $config_type: ident) => {
        fn typename(&self) -> PascalCaseString {
            $typename.try_into().unwrap()
        }

        fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>> {
            let res: $config_type = serde_json::from_value(value.clone())?;
            Ok(Box::new(res))
        }

        fn get_config_schema(&self) -> RootSchema {
            schema_for!($config_type)
        }

        fn get_config_default(&self) -> serde_json::Value {
            serde_json::to_value($config_type::default()).unwrap()
        }
    };
}

pub fn parse_config(config: &JsonReductionConfig) -> Result<Box<dyn ReductionConfig>> {
    match REDUCTION_REGISTRY
        .read()
        .unwrap()
        .get(config.typename.as_ref())
    {
        Some(factory) => factory.parse_config(config.json_value()),
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
    match REDUCTION_REGISTRY
        .read()
        .unwrap()
        .get(config.typename().as_ref())
    {
        Some(factory) => factory.create(config, global_config, num_models_above),
        None => Err(crate::error::Error::InvalidArgument(format!(
            "Unknown reduction type: {}",
            config.typename()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::reduction_factory::JsonReductionConfig;

    #[test]
    fn parse_json_config_with_no_config() {
        let _config: JsonReductionConfig = serde_json::from_value(json!( {
            "typename": "Coin"
        }))
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn parse_json_config_with_no_config_does_not_exist() {
        let _config: JsonReductionConfig = serde_json::from_value(json!( {
            "typename": "DoesNotExist",
        }))
        .unwrap();
    }
}
