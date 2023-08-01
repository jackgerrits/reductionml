use std::{io::BufRead, sync::Arc};

use crate::{
    error::Result, object_pool::Pool, parsers::ParsedFeature,
    sparse_namespaced_features::SparseFeatures, workspace::Workspace, Features, FeaturesType,
    Label, LabelType,
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

    fn create_with_workspace(&self, workspace: &Workspace) -> Self::Parser {
        self.create(
            workspace
                .get_entry_reduction()
                .types()
                .input_features_type(),
            workspace.get_entry_reduction().types().input_label_type(),
            workspace.global_config().hash_seed(),
            workspace.global_config().num_bits(),
            workspace.features_pool().clone(),
        )
    }
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
    fn parse_chunk<'a, 'b>(&self, chunk: &'a str) -> Result<(Features<'b>, Option<Label>)>;
    fn extract_feature_names<'a>(
        &self,
        _chunk: &'a str,
    ) -> Result<std::collections::HashMap<ParsedNamespaceInfo<'a>, Vec<ParsedFeature<'a>>>> {
        todo!()
    }
}
