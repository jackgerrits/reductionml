use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::interactions::{self, Interaction};

#[derive(Serialize, Deserialize, Debug, JsonSchema, DefaultFromSerde)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct GlobalConfig {
    #[serde(default = "default_num_bits")]
    num_bits: u8,

    #[serde(default)]
    hash_seed: u32,

    #[serde(default = "default_true")]
    add_constant_feature: bool,

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
        add_constant_feature: bool,
        interactions: &[Interaction],
    ) -> GlobalConfig {
        GlobalConfig {
            num_bits,
            hash_seed,
            add_constant_feature,
            interactions: interactions.to_vec(),
        }
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    pub fn hash_seed(&self) -> u32 {
        self.hash_seed
    }

    pub fn add_constant_feature(&self) -> bool {
        self.add_constant_feature
    }

    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }
}
