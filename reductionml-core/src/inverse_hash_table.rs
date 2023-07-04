use serde::{Deserialize, Serialize};

use crate::{
    hash::{hash_bytes, FNV_PRIME},
    parsers::ParsedFeature,
    FeatureHash, FeatureIndex,
};

#[derive(Serialize, Deserialize, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub enum Feature {
    // namespace, key
    Simple {
        namespace: String,
        name: String,
    },
    // namespace, key, chain_hashed value
    SimpleWithStringValue {
        namespace: String,
        name: String,
        value: String,
    },
    // namespace, offset
    Anonymous {
        namespace: String,
        offset: u32,
    },
    // An interacted feature should not contain interacted features itself. Maybe this is the wrong way to represent this?
    Interacted {
        terms: Vec<Feature>,
    },
}

impl Feature {
    pub fn hash(&self, hash_seed: u32) -> FeatureHash {
        match &self {
            Feature::Simple { namespace, name } => {
                let namespace_hash = hash_bytes(namespace.as_bytes(), hash_seed);
                hash_bytes(name.as_bytes(), namespace_hash).into()
            }
            Feature::SimpleWithStringValue {
                namespace,
                name,
                value,
            } => {
                let namespace_hash = hash_bytes(namespace.as_bytes(), hash_seed);
                let name_key_hash = hash_bytes(name.as_bytes(), namespace_hash);
                hash_bytes(value.as_bytes(), name_key_hash).into()
            }
            Feature::Anonymous { namespace, offset } => {
                let namespace_hash = hash_bytes(namespace.as_bytes(), hash_seed);
                (namespace_hash + offset).into()
            }
            // In a very cool property hashing of the interacted feature does not need to take into account bit masking until the very end
            // In fact, the produced result is idenitical if interim values are masked or just the final value.
            Feature::Interacted { terms } => {
                let val0 = Feature::hash(terms.first().unwrap(), hash_seed);
                let mut hash_so_far = (FNV_PRIME).wrapping_mul(*val0);
                for term in terms[1..terms.len() - 1].iter() {
                    hash_so_far =
                        (FNV_PRIME).wrapping_mul(hash_so_far ^ *Feature::hash(term, hash_seed));
                }
                hash_so_far ^= *Feature::hash(terms.last().unwrap(), hash_seed);
                hash_so_far.into()
            }
        }
    }

    pub fn from_parsed_feature(parsed_feature: &ParsedFeature, namespace: &str) -> Self {
        match parsed_feature {
            ParsedFeature::Simple { name, .. } => Self::Simple {
                namespace: namespace.to_string(),
                name: name.to_string(),
            },
            ParsedFeature::SimpleWithStringValue { name, value } => Self::SimpleWithStringValue {
                namespace: namespace.to_string(),
                name: name.to_string(),
                value: value.to_string(),
            },
            ParsedFeature::Anonymous { offset, .. } => Self::Anonymous {
                namespace: namespace.to_string(),
                offset: *offset,
            },
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct InverseHashTable {
    hash_table: std::collections::HashMap<FeatureIndex, std::collections::HashSet<Feature>>,
}

impl Default for InverseHashTable {
    fn default() -> Self {
        Self::new()
    }
}

impl InverseHashTable {
    pub fn new() -> Self {
        Self {
            hash_table: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, idx: FeatureIndex, feature: Feature) {
        if let Some(features) = self.hash_table.get_mut(&idx) {
            features.insert(feature);
        } else {
            let mut features = std::collections::HashSet::new();
            features.insert(feature);
            self.hash_table.insert(idx, features);
        }
    }

    pub fn get(&self, idx: FeatureIndex) -> Option<&std::collections::HashSet<Feature>> {
        self.hash_table.get(&idx)
    }
}
