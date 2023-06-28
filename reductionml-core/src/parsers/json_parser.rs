use core::{f32, panic};

use serde::{Deserialize, Serialize};

use crate::error::Result;

use crate::object_pool::Pool;
use crate::parsers::ParsedFeature;
use crate::sparse_namespaced_features::{Namespace, SparseFeatures};
use crate::types::{Features, Label, LabelType};
use crate::{CBAdfFeatures, CBLabel, FeatureIndex, FeatureMask, FeaturesType, SimpleLabel};

use super::{TextModeParser, TextModeParserFactory};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
#[serde(deny_unknown_fields)]
struct SimpleJsonInput<'a> {
    // $schema,
    label: Option<SimpleLabel>,
    features: JsonSparseFeatures<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
#[serde(deny_unknown_fields)]
struct CbJsonInput<'a> {
    // $schema,
    label: Option<CBLabel>,
    features: JsonCBFeatures<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
#[serde(deny_unknown_fields)]
#[serde(untagged)]
enum NamespaceContents<'a> {
    Floats(Vec<f32>),
    Strings(Vec<&'a str>),
    String(&'a str),
    Obj(std::collections::HashMap<&'a str, F<'a>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
#[serde(untagged)]
enum F<'a> {
    Float(f32),
    String(&'a str),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
struct JsonSparseFeatures<'a> {
    #[serde(flatten)]
    namespaces: std::collections::BTreeMap<&'a str, NamespaceContents<'a>>,
}

impl<'a> JsonSparseFeatures<'a> {
    pub fn to_features(
        self,
        mut output: SparseFeatures,
        hash_seed: u32,
        num_bits: u8,
    ) -> SparseFeatures {
        assert!(output.empty());

        for (namespace, contents) in self.namespaces {
            let ns = output.get_or_create_namespace(Namespace::from_name(namespace, hash_seed));
            let ns_hash = ns.namespace().hash(hash_seed);
            let mask = FeatureMask::from_num_bits(num_bits);
            match contents {
                NamespaceContents::Floats(floats) => {
                    let it = (u32::from(ns_hash)..(u32::from(ns_hash) + floats.len() as u32))
                        .map(|x| FeatureIndex::from(x));
                    ns.add_features_with_iter(it, floats.into_iter());
                }
                NamespaceContents::Strings(strings) => {
                    ns.reserve(strings.len());
                    for string in strings {
                        let feat = ParsedFeature::Simple { name: string };
                        ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                    }
                }
                // This might change...
                NamespaceContents::String(string) => {
                    let feat = ParsedFeature::Simple { name: &string };
                    ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                }
                NamespaceContents::Obj(obj) => {
                    ns.reserve(obj.len());
                    for (key, value) in obj {
                        match value {
                            F::Float(float) => {
                                let feat = ParsedFeature::Simple { name: key };
                                ns.add_feature(feat.hash(ns_hash).mask(mask), float);
                            }
                            F::String(string) => {
                                let feat = ParsedFeature::SimpleWithStringValue {
                                    name: key,
                                    value: string,
                                };
                                ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                            }
                            F::Bool(boolean) => {
                                if boolean {
                                    let feat = ParsedFeature::Simple { name: key };
                                    ns.add_feature(feat.hash(ns_hash).mask(mask), 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
struct JsonCBFeatures<'a> {
    shared: Option<JsonSparseFeatures<'a>>,
    actions: Vec<JsonSparseFeatures<'a>>,
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
            _feature_type: features_type,
            _label_type: label_type,
            hash_seed,
            num_bits,
            pool,
        }
    }
}

pub struct JsonParser {
    _feature_type: FeaturesType,
    _label_type: LabelType,
    hash_seed: u32,
    num_bits: u8,
    pool: std::sync::Arc<Pool<SparseFeatures>>,
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
        Ok(match (self._feature_type, self._label_type) {
            (FeaturesType::SparseSimple, LabelType::Simple) => {
                let jd = &mut serde_json::Deserializer::from_str(chunk);

                let result: std::result::Result<SimpleJsonInput, _> =
                    serde_path_to_error::deserialize(jd);
                match &result {
                    Ok(_) => (),
                    Err(err) => {
                        let path = err.path().to_string();
                        panic!("Error parsing json: {:?} at path: {}", err, path);
                    }
                }

                let result = result.unwrap();

                (
                    Features::SparseSimple(result.features.to_features(
                        self.pool.get_object(),
                        self.hash_seed,
                        self.num_bits,
                    )),
                    result.label.map(|x| x.into()),
                )
            }
            (FeaturesType::SparseCBAdf, LabelType::CB) => {
                let jd = &mut serde_json::Deserializer::from_str(chunk);

                let result: std::result::Result<CbJsonInput, _> =
                    serde_path_to_error::deserialize(jd);
                match &result {
                    Ok(_) => (),
                    Err(err) => {
                        let path = err.path().to_string();
                        panic!("Error parsing json: {:?} at path: {}", err, path);
                    }
                }

                let result = result.unwrap();

                (
                    Features::SparseCBAdf(CBAdfFeatures {
                        shared: result.features.shared.map(|features| {
                            features.to_features(
                                self.pool.get_object(),
                                self.hash_seed,
                                self.num_bits,
                            )
                        }),
                        actions: result
                            .features
                            .actions
                            .into_iter()
                            .map(|features| {
                                features.to_features(
                                    self.pool.get_object(),
                                    self.hash_seed,
                                    self.num_bits,
                                )
                            })
                            .collect(),
                    }),
                    result.label.map(|x| x.into()),
                )
            }

            (_, _) => panic!("Feature type mismatch"),
        })
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
        sparse_namespaced_features::Namespace,
        utils::GetInner,
        CBAdfFeatures, CBLabel, FeaturesType, LabelType,
    };
    #[test]
    fn json_parse() {
        let json_obj = json!({
          "label": {
            "action": 3,
            "cost": 0.0,
            "probability": 0.05
          },
          "features": {
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
          }
        });

        let pool = Arc::new(Pool::new());
        let parser = JsonParserFactory::default().create(
            FeaturesType::SparseCBAdf,
            LabelType::CB,
            0,
            18,
            pool,
        );

        let (features, label) = parser.parse_chunk(&json_obj.to_string()).unwrap();
        let cb_label: &CBLabel = label.as_ref().unwrap().get_inner_ref().unwrap();
        assert_eq!(cb_label.action, 3);
        assert_relative_eq!(cb_label.cost, 0.0);
        assert_relative_eq!(cb_label.probability, 0.05);

        let cb_feats: &CBAdfFeatures = features.get_inner_ref().unwrap();
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
