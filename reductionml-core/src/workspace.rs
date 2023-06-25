use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    global_config::GlobalConfig,
    object_pool::Pool,
    reduction::{DepthInfo, ReductionWrapper},
    reduction_factory::JsonReductionConfig,
    sparse_namespaced_features::SparseFeatures,
    types::{Features, Label, Prediction},
};

#[derive(Serialize, Deserialize)]
pub struct Workspace {
    global_config: GlobalConfig,
    entry_reduction: ReductionWrapper,

    #[serde(skip)]
    features_pool: Arc<Pool<SparseFeatures>>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    // $schema,
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    global_config: GlobalConfig,
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    entry_reduction: JsonReductionConfig,
}

impl Workspace {
    pub fn create_from_json(json: &str) -> Result<Workspace> {
        let config: Configuration = serde_json::from_str(json).map_err(|e| {
            Error::InvalidConfiguration(format!("Failed to parse configuration: {e}"))
        })?;

        Self::create_from_configuration(config)
    }

    pub fn create_from_yaml(yaml: &str) -> Result<Workspace> {
        let json_from_yaml = serde_yaml::from_str::<serde_json::Value>(yaml)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to parse yaml: {e}")))?;
        let config: Configuration = serde_json::from_value(json_from_yaml).map_err(|e| {
            Error::InvalidConfiguration(format!("Failed to parse configuration: {e}"))
        })?;

        Self::create_from_configuration(config)
    }

    fn create_from_configuration(config: Configuration) -> Result<Workspace> {
        let reduction_config = crate::reduction_factory::parse_config(&config.entry_reduction)?;
        let entry_reduction = crate::reduction_factory::create_reduction(
            reduction_config.as_ref(),
            &config.global_config,
            1.into(), // Top of the stack must be passed as 1
        )?;

        Ok(Workspace {
            global_config: config.global_config,
            entry_reduction,
            features_pool: Arc::new(Pool::new()),
        })
    }

    // TODO move to bincode or msgpack
    pub fn create_from_model(json: &[u8]) -> Result<Workspace> {
        let r = flexbuffers::Reader::get_root(json).unwrap();
        Ok(Workspace::deserialize(r).unwrap())
    }

    // TODO move to bincode or msgpack
    pub fn serialize_model(&self) -> Result<Vec<u8>> {
        let mut s = flexbuffers::FlexbufferSerializer::new();
        self.serialize(&mut s).unwrap();
        Ok(s.take_buffer())
    }

    // experimental
    pub fn serialize_to_json(&self) -> Result<String> {
        // TODO: rewrite weights to hide the internal index details
        serde_json::to_string(&self)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to serialize model: {e}")))
    }

    // experimental
    pub fn deserialize_from_json(json: &str) -> Result<Workspace> {
        // TODO: convert back weights to hide the internal index details
        serde_json::from_str(json)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to parse model: {e}")))
    }

    pub fn predict(&self, features: &Features) -> Prediction {
        let mut depth_info = DepthInfo::new();
        self.entry_reduction
            .predict(features, &mut depth_info, 0.into())
    }

    pub fn predict_then_learn(&mut self, features: &Features, label: &Label) -> Prediction {
        let mut depth_info = DepthInfo::new();
        self.entry_reduction
            .predict_then_learn(features, label, &mut depth_info, 0.into())
    }

    pub fn learn(&mut self, features: &Features, label: &Label) {
        let mut depth_info = DepthInfo::new();
        self.entry_reduction
            .learn(features, label, &mut depth_info, 0.into());
    }

    pub fn get_entry_reduction(&self) -> &ReductionWrapper {
        &self.entry_reduction
    }

    pub fn global_config(&self) -> &GlobalConfig {
        &self.global_config
    }

    pub fn features_pool(&self) -> &Arc<Pool<SparseFeatures>> {
        &self.features_pool
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use serde_json::json;

    use crate::{sparse_namespaced_features::SparseFeatures, utils::GetInner, ScalarPrediction};

    use super::*;

    #[test]
    fn test_create_workspace() {
        let config = json!(
            {
                "globalConfig": {
                    "numBits": 4
                },
                "entryReduction": {
                    "typename": "Coin",
                    "config": {
                        "alpha": 10
                    }
                }
            }
        );

        let mut workspace = Workspace::create_from_json(&config.to_string()).unwrap();

        let mut features = SparseFeatures::new();
        let ns =
            features.get_or_create_namespace(crate::sparse_namespaced_features::Namespace::Default);
        ns.add_feature(0.into(), 1.0);
        ns.add_feature(2.into(), 1.0);
        ns.add_feature(3.into(), 1.0);
        features.add_constant_feature(4);

        let features = Features::SparseSimple(features);

        let pred = workspace.predict(&features);
        let scalar_pred: &ScalarPrediction = pred.get_inner_ref().unwrap();
        assert_relative_eq!(scalar_pred.prediction, 0.0);

        let label = Label::Simple(0.5.into());

        // For some reason two calls to learn are required to get a non-zero prediction for coin?
        workspace.learn(&features, &label);
        workspace.learn(&features, &label);

        let pred = workspace.predict(&features);
        let scalar_pred: &ScalarPrediction = pred.get_inner_ref().unwrap();
        assert_relative_eq!(scalar_pred.prediction, 0.5);
    }
}
