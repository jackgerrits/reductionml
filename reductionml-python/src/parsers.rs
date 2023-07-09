use std::sync::Arc;

use pyo3::prelude::*;
use reductionml_core::{
    parsers::{TextModeParser, TextModeParserFactory},
    FeaturesType, LabelType,
};

use crate::{
    features::WrappedFeaturesForReturn, labels::WrappedLabel, WrappedError, SPARSE_FEATURES_POOL,
};

#[pyclass]
#[pyo3(name = "Parser")]
pub(crate) struct WrappedParser(Arc<dyn reductionml_core::parsers::TextModeParser>);

unsafe impl Send for WrappedParser {}
unsafe impl Sync for WrappedParser {}

#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormatType {
    VwText,
    Json,
    DsJson,
}

#[pyclass]
#[derive(Clone, Copy)]
pub(crate) enum ReductionType {
    Simple,
    CB,
}

impl From<ReductionType> for (reductionml_core::FeaturesType, reductionml_core::LabelType) {
    fn from(x: ReductionType) -> Self {
        match x {
            ReductionType::Simple => (FeaturesType::SparseSimple, LabelType::Simple),
            ReductionType::CB => (FeaturesType::SparseCBAdf, LabelType::CB),
        }
    }
}

impl FormatType {
    pub(crate) fn get_parser(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
    ) -> Box<dyn TextModeParser> {
        match self {
            FormatType::VwText => Box::new(
                reductionml_core::parsers::VwTextParserFactory::default().create(
                    features_type,
                    label_type,
                    hash_seed,
                    num_bits,
                    SPARSE_FEATURES_POOL.clone(),
                ),
            ),
            FormatType::DsJson => Box::new(
                reductionml_core::parsers::DsJsonParserFactory::default().create(
                    features_type,
                    label_type,
                    hash_seed,
                    num_bits,
                    SPARSE_FEATURES_POOL.clone(),
                ),
            ),
            FormatType::Json => Box::new(
                reductionml_core::parsers::JsonParserFactory::default().create(
                    features_type,
                    label_type,
                    hash_seed,
                    num_bits,
                    SPARSE_FEATURES_POOL.clone(),
                ),
            ),
        }
    }
}

#[pymethods]
impl WrappedParser {
    fn parse(
        &self,
        chunk: &str,
    ) -> Result<(WrappedFeaturesForReturn, Option<WrappedLabel>), PyErr> {
        let (feats, label) = self
            .0
            .parse_chunk(chunk)
            .map_err(|x| WrappedError::from(x))?;
        let feats: WrappedFeaturesForReturn =
            feats.try_into().map_err(|x| WrappedError::from(x))?;
        let label: Option<WrappedLabel> = label.map(|x| x.into());
        Ok((feats, label))
    }

    #[staticmethod]
    fn create_parser(
        format_type: FormatType,
        reduction_type: ReductionType,
        hash_seed: u32,
        num_bits: u8,
    ) -> Result<WrappedParser, PyErr> {
        let (features_type, label_type) = reduction_type.into();
        let parser = format_type.get_parser(features_type, label_type, hash_seed, num_bits);
        Ok(WrappedParser(parser.into()))
    }

    #[staticmethod]
    fn create_parser_with_workspace(
        format_type: FormatType,
        workspace: &crate::workspace::WrappedWorkspace,
    ) -> Result<WrappedParser, PyErr> {
        let features_type = workspace
            .0
            .get_entry_reduction()
            .types()
            .input_features_type();
        let label_type = workspace.0.get_entry_reduction().types().input_label_type();
        let hash_seed = workspace.0.global_config().hash_seed();
        let num_bits = workspace.0.global_config().num_bits();
        let parser = format_type.get_parser(features_type, label_type, hash_seed, num_bits);
        Ok(WrappedParser(parser.into()))
    }
}
