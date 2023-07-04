use core::f32;

use serde_json_borrow::Value;

use crate::error::Result;

use crate::object_pool::Pool;
use crate::parsers::ParsedFeature;
use crate::sparse_namespaced_features::{Namespace, SparseFeatures};
use crate::types::{Features, Label, LabelType};
use crate::{CBAdfFeatures, CBLabel, FeatureMask, FeaturesType};

use super::{ParsedNamespaceInfo, TextModeParser, TextModeParserFactory};

#[derive(Default)]
pub struct DsJsonParserFactory;
impl TextModeParserFactory for DsJsonParserFactory {
    type Parser = DsJsonParser;

    fn create(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
        pool: std::sync::Arc<Pool<SparseFeatures>>,
    ) -> DsJsonParser {
        // Only supports CB
        if features_type != FeaturesType::SparseCBAdf {
            panic!("DsJsonParser only supports SparseCBAdf")
        }

        if label_type != LabelType::CB {
            panic!("DsJsonParser only supports CB labels")
        }

        DsJsonParser {
            _feature_type: features_type,
            _label_type: label_type,
            hash_seed,
            num_bits,
            pool,
        }
    }
}

pub struct DsJsonParser {
    _feature_type: FeaturesType,
    _label_type: LabelType,
    hash_seed: u32,
    num_bits: u8,
    pool: std::sync::Arc<Pool<SparseFeatures>>,
}

impl DsJsonParser {
    pub fn handle_features(
        &self,
        features: &mut SparseFeatures,
        object_key: &str,
        json_value: &Value,
        namespace_stack: &mut Vec<Namespace>,
    ) {
        // All underscore prefixed keys are ignored.
        if object_key.starts_with('_') {
            return;
        }

        // skip everything with _
        match json_value {
            Value::Null => panic!("Null is not supported"),
            Value::Bool(true) => {
                let current_ns = *namespace_stack
                    .last()
                    .expect("namespace stack should not be empty here");
                let current_ns_hash = current_ns.hash(self.hash_seed);
                let current_feats = features.get_or_create_namespace(current_ns);
                current_feats.add_feature(
                    ParsedFeature::Simple { name: object_key }
                        .hash(current_ns_hash)
                        .mask(FeatureMask::from_num_bits(self.num_bits)),
                    1.0,
                );
            }
            Value::Bool(false) => (),
            Value::Number(value) => {
                let current_ns = *namespace_stack
                    .last()
                    .expect("namespace stack should not be empty here");
                let current_ns_hash = current_ns.hash(self.hash_seed);
                let current_feats = features.get_or_create_namespace(current_ns);
                current_feats.add_feature(
                    ParsedFeature::Simple { name: object_key }
                        .hash(current_ns_hash)
                        .mask(FeatureMask::from_num_bits(self.num_bits)),
                    value.as_f64().unwrap() as f32,
                );
            }
            Value::Str(value) => {
                let current_ns = namespace_stack
                    .last()
                    .expect("namespace stack should not be empty here");
                let current_ns_hash = current_ns.hash(self.hash_seed);
                let current_feats = features.get_or_create_namespace(*current_ns);
                current_feats.add_feature(
                    ParsedFeature::SimpleWithStringValue {
                        name: object_key,
                        value,
                    }
                    .hash(current_ns_hash)
                    .mask(FeatureMask::from_num_bits(self.num_bits)),
                    1.0,
                );
            }
            Value::Array(value) => {
                namespace_stack.push(Namespace::from_name(object_key, self.hash_seed));
                let current_ns = *namespace_stack
                    .last()
                    .expect("namespace stack should not be empty here");
                let current_ns_hash = current_ns.hash(self.hash_seed);
                for (anon_idx, v) in value.iter().enumerate() {
                    match v {
                        Value::Number(value) => {
                            // Not super efficient but it works
                            // Doing this in the outside doesn't work as it would mean two mutable borrows.
                            let current_feats = features.get_or_create_namespace(current_ns);
                            current_feats.add_feature(
                                ParsedFeature::Anonymous {
                                    offset: anon_idx as u32,
                                }
                                .hash(current_ns_hash)
                                .mask(FeatureMask::from_num_bits(self.num_bits)),
                                value.as_f64().unwrap() as f32,
                            );
                        }
                        Value::Object(_) => {
                            self.handle_features(features, object_key, v, namespace_stack);
                        }
                        // Just ignore null and do nothing
                        Value::Null => (),
                        _ => panic!(
                            "Array of non-number or object is not supported key:{} value:{:?}",
                            object_key, v
                        ),
                    }
                }
                namespace_stack.pop().unwrap();
            }
            Value::Object(value) => {
                namespace_stack.push(Namespace::from_name(object_key, self.hash_seed));
                for (key, v) in value {
                    self.handle_features(features, key, v, namespace_stack);
                }
                namespace_stack.pop().unwrap();
            }
        }
    }
}

impl TextModeParser for DsJsonParser {
    fn get_next_chunk(
        &self,
        input: &mut dyn std::io::BufRead,
        mut output_buffer: String,
    ) -> Result<Option<String>> {
        output_buffer.clear();
        input.read_line(&mut output_buffer)?;
        if output_buffer.is_empty() {
            return Ok(None);
        }
        Ok(Some(output_buffer))
    }

    fn parse_chunk<'a, 'b>(&self, chunk: &'a str) -> Result<(Features<'b>, Option<Label>)> {
        let json: Value = serde_json::from_str(chunk).expect("JSON was not well-formatted");

        let mut namespace_stack = Vec::new();

        let mut shared_ex = self.pool.get_object();
        self.handle_features(&mut shared_ex, " ", json.get("c"), &mut namespace_stack);
        assert!(namespace_stack.is_empty());

        let mut actions = Vec::new();
        for item in json.get("c").get("_multi").iter_array().unwrap() {
            let mut action = self.pool.get_object();
            self.handle_features(&mut action, " ", item, &mut namespace_stack);
            actions.push(action);
            assert!(namespace_stack.is_empty());
        }

        let label = match (
            json.get("_label_cost"),
            json.get("_label_probability"),
            json.get("_labelIndex"),
        ) {
            (Value::Number(cost), Value::Number(prob), Value::Number(action)) => Some(CBLabel {
                action: action.as_u64().unwrap() as usize,
                cost: cost.as_f64().unwrap() as f32,
                probability: prob.as_f64().unwrap() as f32,
            }),
            (Value::Null, Value::Null, Value::Null) => None,
            _ => panic!("Invalid label, all 3 or none must be present"),
        };

        Ok((
            Features::SparseCBAdf(CBAdfFeatures {
                shared: Some(shared_ex),
                actions,
            }),
            label.map(Label::CB),
        ))
    }

    fn extract_feature_names<'a>(
        &self,
        _chunk: &'a str,
    ) -> Result<std::collections::HashMap<ParsedNamespaceInfo<'a>, Vec<ParsedFeature<'a>>>> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use serde_json::json;

    use crate::{
        object_pool::Pool,
        parsers::{DsJsonParserFactory, TextModeParser, TextModeParserFactory},
        sparse_namespaced_features::Namespace,
        utils::AsInner,
        CBAdfFeatures, CBLabel, FeaturesType, LabelType,
    };
    #[test]
    fn extract_dsjson_test_chain_hash() {
        let json_obj = json!({
          "_label_cost": -0.0,
          "_label_probability": 0.05000000074505806,
          "_label_Action": 4,
          "_labelIndex": 3,
          "o": [
            {
              "v": 0.0,
              "EventId": "13118d9b4c114f8485d9dec417e3aefe",
              "ActionTaken": false
            }
          ],
          "Timestamp": "2021-02-04T16:31:29.2460000Z",
          "Version": "1",
          "EventId": "13118d9b4c114f8485d9dec417e3aefe",
          "a": [4, 2, 1, 3],
          "c": {
            "bool_true": true,
            "bool_false": false,
            "numbers": [4, 5.6],
            "FromUrl": [
              { "timeofday": "Afternoon", "weather": "Sunny", "name": "Cathy" }
            ],
            "_multi": [
              {
                "_tag": "Cappucino",
                "i": { "constant": 1, "id": "Cappucino" },
                "j": [
                  {
                    "type": "hot",
                    "origin": "kenya",
                    "organic": "yes",
                    "roast": "dark"
                  }
                ]
              }
            ]
          },
          "p": [0.05, 0.05, 0.05, 0.85],
          "VWState": {
            "m": "ff0744c1aa494e1ab39ba0c78d048146/550c12cbd3aa47f09fbed3387fb9c6ec"
          },
          "_original_label_cost": -0.0
        });

        let pool = Arc::new(Pool::new());
        let parser = DsJsonParserFactory::default().create(
            FeaturesType::SparseCBAdf,
            LabelType::CB,
            0,
            18,
            pool,
        );

        let (features, label) = parser.parse_chunk(&json_obj.to_string()).unwrap();
        let cb_label: &CBLabel = label.as_ref().unwrap().as_inner().unwrap();
        assert_eq!(cb_label.action, 3);
        assert_relative_eq!(cb_label.cost, 0.0);
        assert_relative_eq!(cb_label.probability, 0.05);

        let cb_feats: &CBAdfFeatures = features.as_inner().unwrap();
        assert_eq!(cb_feats.actions.len(), 1);
        assert!(cb_feats.shared.is_some());
        let shared = cb_feats.shared.as_ref().unwrap();
        assert_eq!(shared.namespaces().count(), 3);
        let shared_default_ns = shared.get_namespace(Namespace::Default).unwrap();
        assert_eq!(shared_default_ns.iter().count(), 1);

        let shared_from_url_ns = shared
            .get_namespace(Namespace::from_name("FromUrl", 0))
            .unwrap();
        assert_eq!(shared_from_url_ns.iter().count(), 3);

        let shared_numbers_ns = shared
            .get_namespace(Namespace::from_name("numbers", 0))
            .unwrap();
        assert_eq!(shared_numbers_ns.iter().count(), 2);
        assert_relative_eq!(
            shared_numbers_ns.iter().map(|(_, val)| val).sum::<f32>(),
            9.6
        );

        let action = cb_feats.actions.get(0).unwrap();
        assert_eq!(action.namespaces().count(), 2);
        assert!(action.get_namespace(Namespace::Default).is_none());
        let action_i_ns = action.get_namespace(Namespace::from_name("i", 0)).unwrap();
        assert_eq!(action_i_ns.iter().count(), 2);
        let action_j_ns = action.get_namespace(Namespace::from_name("j", 0)).unwrap();
        assert_eq!(action_j_ns.iter().count(), 4);
    }
}
