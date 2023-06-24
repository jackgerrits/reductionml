mod text_parser;
use std::io::Cursor;

use murmur3::murmur3_32;
pub use text_parser::*;
mod vw_text_parser;
pub use vw_text_parser::*;
mod dsjson_parser;
pub use dsjson_parser::*;

use crate::{FeatureHash, NamespaceHash};

pub enum ParsedFeature<'a> {
    Simple { name: &'a str },
    SimpleWithStringValue { name: &'a str, value: &'a str },
    Anonymous { offset: u32 },
}

impl<'a> ParsedFeature<'a> {
    pub fn hash(&self, namespace_hash: NamespaceHash) -> FeatureHash {
        match &self {
            ParsedFeature::Simple { name } => {
                FeatureHash::from(murmur3_32(&mut Cursor::new(name), *namespace_hash).unwrap())
            }
            ParsedFeature::SimpleWithStringValue { name, value } => {
                let name_key_hash = murmur3_32(&mut Cursor::new(name), *namespace_hash).unwrap();
                FeatureHash::from(murmur3_32(&mut Cursor::new(value), name_key_hash).unwrap())
            }
            ParsedFeature::Anonymous { offset } => (*namespace_hash + offset).into(),
        }
    }
}
