use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dense_weights::{DenseWeights, DenseWeightsWithNDArray},
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

impl Configuration {
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str::<serde_json::Value>(json)?.try_into()
    }

    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml::from_str::<serde_yaml::Value>(yaml)?.try_into()
    }
}

impl TryInto<Configuration> for serde_json::Value {
    type Error = Error;

    fn try_into(self) -> std::result::Result<Configuration, Self::Error> {
        serde_json::from_value(self)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to parse configuration: {e}")))
    }
}

impl TryInto<Configuration> for serde_yaml::Value {
    type Error = Error;

    fn try_into(self) -> std::result::Result<Configuration, Self::Error> {
        serde_yaml::from_value::<serde_json::Value>(self)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to parse yaml: {e}")))?
            .try_into()
    }
}

// We need to search until we find an object with the keys weights, feature_index_size, model_index_size, feature_state_size, model_index_size_shift, feature_state_size_shift
fn rewrite_json_ndarray_to_sparse(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if map.contains_key("weights")
                && map.contains_key("feature_index_size")
                && map.contains_key("model_index_size")
                && map.contains_key("feature_state_size")
                && map.contains_key("model_index_size_shift")
                && map.contains_key("feature_state_size_shift")
            {
                let wts: DenseWeightsWithNDArray = serde_json::from_value(value.clone()).unwrap();
                *value = serde_json::to_value(wts.to_dense_weights()).unwrap();
                return;
            }
            for (_, v) in map {
                rewrite_json_ndarray_to_sparse(v);
            }
        }
        serde_json::Value::Array(vec) => {
            for v in vec {
                rewrite_json_ndarray_to_sparse(v);
            }
        }
        _ => (),
    }
}

fn rewrite_json_sparse_to_ndarray(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if map.contains_key("weights")
                && map.contains_key("feature_index_size")
                && map.contains_key("model_index_size")
                && map.contains_key("feature_state_size")
                && map.contains_key("model_index_size_shift")
                && map.contains_key("feature_state_size_shift")
            {
                let wts: DenseWeights = serde_json::from_value(value.clone()).unwrap();
                *value =
                    serde_json::to_value(DenseWeightsWithNDArray::from_dense_weights(wts)).unwrap();
                return;
            }
            for (_, v) in map {
                rewrite_json_sparse_to_ndarray(v);
            }
        }
        serde_json::Value::Array(vec) => {
            for v in vec {
                rewrite_json_sparse_to_ndarray(v);
            }
        }
        _ => (),
    }
}

impl Workspace {
    pub fn new(config: Configuration) -> Result<Workspace> {
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
    pub fn serialize_to_json(&self) -> Result<Value> {
        let mut value = serde_json::to_value(self).unwrap();
        rewrite_json_sparse_to_ndarray(&mut value);
        Ok(value)
    }

    // experimental
    pub fn deserialize_from_json(json: &Value) -> Result<Workspace> {
        let mut value: serde_json::Value = serde_json::from_value(json.clone()).map_err(|e| {
            Error::InvalidConfiguration(format!("Failed to parse configuration: {e}"))
        })?;
        rewrite_json_ndarray_to_sparse(&mut value);
        serde_json::from_value(value)
            .map_err(|e| Error::InvalidConfiguration(format!("Failed to parse model: {e}")))
    }

    pub fn predict(&self, features: &mut Features) -> Prediction {
        let mut depth_info = DepthInfo::new();
        self.entry_reduction
            .predict(features, &mut depth_info, 0.into())
    }

    pub fn predict_then_learn(&mut self, features: &mut Features, label: &Label) -> Prediction {
        let mut depth_info = DepthInfo::new();
        self.entry_reduction
            .predict_then_learn(features, label, &mut depth_info, 0.into())
    }

    pub fn learn(&mut self, features: &mut Features, label: &Label) {
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

    use crate::{sparse_namespaced_features::SparseFeatures, utils::AsInner, ScalarPrediction};

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

        let mut workspace = Workspace::new(config.try_into().unwrap()).unwrap();

        let mut features = SparseFeatures::new();
        let ns =
            features.get_or_create_namespace(crate::sparse_namespaced_features::Namespace::Default);
        ns.add_feature(0.into(), 1.0);
        ns.add_feature(2.into(), 1.0);
        ns.add_feature(3.into(), 1.0);

        let mut features = Features::SparseSimple(features);

        let pred = workspace.predict(&mut features);
        let scalar_pred: &ScalarPrediction = pred.as_inner().unwrap();
        assert_relative_eq!(scalar_pred.prediction, 0.0);

        let label = Label::Simple(0.5.into());

        // For some reason two calls to learn are required to get a non-zero prediction for coin?
        workspace.learn(&mut features, &label);
        workspace.learn(&mut features, &label);

        let pred = workspace.predict(&mut features);
        let scalar_pred: &ScalarPrediction = pred.as_inner().unwrap();
        assert_relative_eq!(scalar_pred.prediction, 0.5);
    }
}
