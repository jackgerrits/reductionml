mod text_parser;

pub use text_parser::*;
mod vw_text_parser;
pub use vw_text_parser::*;
mod dsjson_parser;
pub use dsjson_parser::*;
mod json_parser;
pub use json_parser::*;

use crate::{hash::hash_bytes, FeatureHash, NamespaceHash};

pub enum ParsedFeature<'a> {
    Simple { name: &'a str },
    SimpleWithStringValue { name: &'a str, value: &'a str },
    Anonymous { offset: u32 },
}

impl<'a> ParsedFeature<'a> {
    pub fn hash(&self, namespace_hash: NamespaceHash) -> FeatureHash {
        match &self {
            ParsedFeature::Simple { name } => {
                FeatureHash::from(hash_bytes(name.as_bytes(), *namespace_hash))
            }
            ParsedFeature::SimpleWithStringValue { name, value } => {
                let name_key_hash = hash_bytes(name.as_bytes(), *namespace_hash);
                FeatureHash::from(hash_bytes(value.as_bytes(), name_key_hash))
            }
            ParsedFeature::Anonymous { offset } => (*namespace_hash + offset).into(),
        }
    }
}
