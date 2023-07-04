use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::interactions::Interaction;

#[derive(Serialize, Deserialize, Debug, JsonSchema, DefaultFromSerde)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct GlobalConfig {
    #[serde(default = "default_num_bits")]
    num_bits: u8,

    #[serde(default)]
    hash_seed: u32,

    #[serde(default = "default_true")]
    constant_feature_enabled: bool,

    #[serde(default)]
    interactions: Vec<Interaction>,
}

fn default_num_bits() -> u8 {
    18
}

fn default_true() -> bool {
    true
}

impl GlobalConfig {
    pub fn new(
        num_bits: u8,
        hash_seed: u32,
        constant_feature_enabled: bool,
        interactions: &[Interaction],
    ) -> GlobalConfig {
        GlobalConfig {
            num_bits,
            hash_seed,
            constant_feature_enabled,
            interactions: interactions.to_vec(),
        }
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    pub fn hash_seed(&self) -> u32 {
        self.hash_seed
    }

    pub fn constant_feature_enabled(&self) -> bool {
        self.constant_feature_enabled
    }

    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }
}
