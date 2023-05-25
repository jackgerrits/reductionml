mod text_parser;
pub use text_parser::*;
mod vw_text_parser;
pub use vw_text_parser::*;

use crate::{hash::murmurhash3_32, FeatureHash, NamespaceHash};

pub enum ParsedFeature<'a> {
    Simple { name: &'a str },
    SimpleWithStringValue { name: &'a str, value: &'a str },
    Anonymous { offset: u32 },
}

impl<'a> ParsedFeature<'a> {
    pub fn hash(&self, namespace_hash: NamespaceHash) -> FeatureHash {
        match &self {
            ParsedFeature::Simple { name } => {
                FeatureHash::from(murmurhash3_32(name.as_bytes(), *namespace_hash))
            }
            ParsedFeature::SimpleWithStringValue { name, value } => {
                let name_key_hash = murmurhash3_32(name.as_bytes(), *namespace_hash);
                FeatureHash::from(murmurhash3_32(value.as_bytes(), name_key_hash))
            }
            ParsedFeature::Anonymous { offset } => (*namespace_hash + offset).into(),
        }
    }
}
