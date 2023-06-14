use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct GlobalConfig {
    num_bits: u8,
}

impl GlobalConfig {
    pub fn new(num_bits: u8) -> GlobalConfig {
        GlobalConfig { num_bits }
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }
}
