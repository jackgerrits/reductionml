use core::{f32, panic};

use crate::error::Result;

use crate::object_pool::Pool;
use crate::parsers::ParsedFeature;
use crate::sparse_namespaced_features::{Namespace, SparseFeatures};
use crate::types::{Features, Label, LabelType};
use crate::{CBAdfFeatures, CBLabel, FeatureHash, FeatureMask, FeaturesType, SimpleLabel};

use super::{TextModeParser, TextModeParserFactory};

use serde_json_borrow::Value;

pub fn to_features(
    val: &Value,
    mut output: SparseFeatures,
    hash_seed: u32,
    num_bits: u8,
) -> SparseFeatures {
    match val {
        Value::Object(obj) => {
            for (ns_name, value) in obj {
                let ns = output.get_or_create_namespace(Namespace::from_name(ns_name, hash_seed));
                let ns_hash = ns.namespace().hash(hash_seed);
                let mask = FeatureMask::from_num_bits(num_bits);
                match value {
                    Value::Str(_) => todo!(),
                    Value::Array(ar) => match ar.first() {
                        Some(Value::Number(_)) => {
                            let it = (u32::from(ns_hash)..(u32::from(ns_hash) + ar.len() as u32))
                                .map(|x| FeatureHash::from(x).mask(mask));
                            ns.add_features_with_iter(
                                it,
                                ar.iter().map(|x| {
                                    x.as_f64().expect("Arrays must contain the same type") as f32
                                }),
                            );
                        }
                        Some(Value::Str(_)) => {
                            ns.reserve(ar.len());
                            for string in ar {
                                let feat = ParsedFeature::Simple {
                                    name: string
                                        .as_str()
                                        .expect("Arrays must contain the same type"),
                                };
                                ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                            }
                        }
                        Some(_) => panic!("Not a number or string"),
                        None => todo!(),
                    },

                    Value::Object(contents) => {
                        for (key, value) in contents {
                            match value {
                                Value::Number(value) => {
                                    let feat: ParsedFeature<'_> =
                                        ParsedFeature::Simple { name: key };
                                    ns.add_feature(
                                        feat.hash(ns_hash).mask(mask),
                                        value.as_f64().unwrap() as f32,
                                    );
                                }
                                Value::Str(value) => {
                                    let feat =
                                        ParsedFeature::SimpleWithStringValue { name: key, value };
                                    ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                                }
                                Value::Bool(value) => {
                                    if *value {
                                        let feat = ParsedFeature::Simple { name: key };
                                        ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                                    }
                                }
                                _ => todo!(),
                            }
                        }
                    }
                    _ => todo!(),
                }
            }
        }
        _ => panic!("Not an object"),
    }
    output
}

#[derive(Default)]
pub struct JsonParserFactory;
impl TextModeParserFactory for JsonParserFactory {
    type Parser = JsonParser;

    fn create(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
        pool: std::sync::Arc<Pool<SparseFeatures>>,
    ) -> JsonParser {
        JsonParser {
            features_type,
            label_type,
            hash_seed,
            num_bits,
            pool,
        }
    }
}

pub struct JsonParser {
    features_type: FeaturesType,
    label_type: LabelType,
    hash_seed: u32,
    num_bits: u8,
    pool: std::sync::Arc<Pool<SparseFeatures>>,
}

impl<'a, 'b> JsonParser {
    fn parse_features(&self, json_input: &'a Value) -> Result<Features<'b>> {
        match self.features_type {
            FeaturesType::SparseSimple => match json_input.get("features") {
                Value::Null => panic!("No features found"),
                val => Ok(
                    to_features(val, self.pool.get_object(), self.hash_seed, self.num_bits).into(),
                ),
            },
            FeaturesType::SparseCBAdf => {
                let shared = match json_input.get("shared") {
                    Value::Null => None,
                    val => {
                        let feats =
                            to_features(val, self.pool.get_object(), self.hash_seed, self.num_bits);
                        Some(feats)
                    }
                };

                let actions = match json_input.get("actions") {
                    Value::Null => panic!("No actions found"),
                    Value::Array(val) => val
                        .iter()
                        .map(|x| {
                            to_features(x, self.pool.get_object(), self.hash_seed, self.num_bits)
                        })
                        .collect(),
                    _ => panic!("Actions must be an array"),
                };

                Ok(CBAdfFeatures { shared, actions }.into())
            }
        }
    }
}

impl JsonParser {
    fn parse_label(&self, json_input: &Value) -> Result<Option<Label>> {
        match self.label_type {
            LabelType::Simple => Ok(match json_input.get("label") {
                Value::Null => None,
                Value::Number(val) => Some(SimpleLabel::from(val.as_f64().unwrap() as f32)),
                val => {
                    let l: SimpleLabel =
                        serde_json::from_value(serde_json::Value::from(val.clone())).unwrap();
                    Some(l)
                }
            }
            .map(Into::into)),
            LabelType::Binary => todo!(),
            LabelType::CB => Ok(match json_input.get("label") {
                Value::Null => None,
                val => {
                    let l: CBLabel =
                        serde_json::from_value(serde_json::Value::from(val.clone())).unwrap();
                    Some(l.into())
                }
            }),
        }
    }
}

impl TextModeParser for JsonParser {
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
        let features = self.parse_features(&json)?;
        let label = self.parse_label(&json)?;
        Ok((features, label))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use serde_json::json;

    use crate::{
        object_pool::Pool,
        parsers::{JsonParserFactory, TextModeParser, TextModeParserFactory},
        sparse_namespaced_features::{Namespace, SparseFeatures},
        utils::AsInner,
        CBAdfFeatures, CBLabel, FeaturesType, LabelType, SimpleLabel,
    };
    #[test]
    fn json_parse_cb() {
        let json_obj = json!({
            "label": {
                "action": 3,
                "cost": 0.0,
                "probability": 0.05
              },
            "shared": {
                ":default": {
                    "bool_true": true,
                    "bool_false": false
                },
                "numbers": [4, 5.6],
                "FromUrl": {
                    "timeofday": "Afternoon",
                    "weather": "Sunny",
                    "name": "Cathy"
                }
            },
            "actions": [
                {
                "i": { "constant": 1, "id": "Cappucino" },
                "j": {
                    "type": "hot",
                    "origin": "kenya",
                    "organic": "yes",
                    "roast": "dark"
                }
                }
            ]
        });

        let pool = Arc::new(Pool::new());
        let parser = JsonParserFactory::default().create(
            FeaturesType::SparseCBAdf,
            LabelType::CB,
            0,
            18,
            pool,
        );

        let input = json_obj.to_string();
        let (features, label) = parser.parse_chunk(&input).unwrap();
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

    #[test]
    fn json_parse_simple() {
        let json_obj = json!({
            "label": {
                "value": 0.2,
                "weight": 0.4
            },
            "features" : {
                ":default": {
                    "bool_true": true,
                    "bool_false": false
                },
                "numbers": [4, 5.6],
                "FromUrl": {
                    "timeofday": "Afternoon",
                    "weather": "Sunny",
                    "name": "Cathy"
                }
            }
        });

        let pool = Arc::new(Pool::new());
        let parser = JsonParserFactory::default().create(
            FeaturesType::SparseSimple,
            LabelType::Simple,
            0,
            18,
            pool,
        );

        let input = json_obj.to_string();
        let (features, label) = parser.parse_chunk(&input).unwrap();
        let lbl: &SimpleLabel = label.as_ref().unwrap().as_inner().unwrap();
        assert_relative_eq!(lbl.value(), 0.2);
        assert_relative_eq!(lbl.weight(), 0.4);

        let features: &SparseFeatures = features.as_inner().unwrap();
        assert_eq!(features.namespaces().count(), 3);
        let features_default_ns = features.get_namespace(Namespace::Default).unwrap();
        assert_eq!(features_default_ns.iter().count(), 1);

        let features_from_url_ns = features
            .get_namespace(Namespace::from_name("FromUrl", 0))
            .unwrap();
        assert_eq!(features_from_url_ns.iter().count(), 3);

        let features_numbers_ns = features
            .get_namespace(Namespace::from_name("numbers", 0))
            .unwrap();
        assert_eq!(features_numbers_ns.iter().count(), 2);
        assert_relative_eq!(
            features_numbers_ns.iter().map(|(_, val)| val).sum::<f32>(),
            9.6
        );
    }
}
