use std::{io::BufRead, sync::Arc};

use crate::{
    error::Result, object_pool::Pool, parsers::ParsedFeature,
    sparse_namespaced_features::SparseFeatures, Features, FeaturesType, Label, LabelType,
};

pub trait TextModeParserFactory {
    type Parser: TextModeParser;
    fn create(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
        pool: Arc<Pool<SparseFeatures>>,
    ) -> Self::Parser;
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ParsedNamespaceInfo<'a> {
    Named(&'a str),
    Default,
}

pub trait TextModeParser: Sync {
    fn get_next_chunk(
        &self,
        input: &mut dyn BufRead,
        output_buffer: String,
    ) -> Result<Option<String>>;
    fn parse_chunk<'a>(&self, chunk: &str) -> Result<(Features<'a>, Option<Label>)>;
    fn extract_feature_names<'a>(
        &self,
        chunk: &'a str,
    ) -> Result<std::collections::HashMap<ParsedNamespaceInfo<'a>, Vec<ParsedFeature<'a>>>> {
        todo!()
    }
}
