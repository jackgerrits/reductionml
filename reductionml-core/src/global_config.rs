use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

#[derive(Serialize, Deserialize, Debug, JsonSchema, DefaultFromSerde)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct GlobalConfig {
    #[serde(default = "default_num_bits")]
    num_bits: u8,

    #[serde(default)]
    hash_seed: u32,
}

fn default_num_bits() -> u8 {
    18
}

impl GlobalConfig {
    pub fn new(num_bits: u8, hash_seed: u32) -> GlobalConfig {
        GlobalConfig {
            num_bits,
            hash_seed,
        }
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    pub fn hash_seed(&self) -> u32 {
        self.hash_seed
    }
}
