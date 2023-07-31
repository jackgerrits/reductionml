use std::{borrow::Cow, sync::Arc};

use pyo3::{prelude::*, types::PyDict};
use pythonize::depythonize;
use reductionml_core::{
    parsers::{TextModeParser, TextModeParserFactory},
    FeaturesType, LabelType,
};

use crate::{
    features::WrappedFeaturesForReturn, labels::{WrappedLabel, WrappedLabelType}, WrappedError, SPARSE_FEATURES_POOL,
    WrappedFeaturesType
};

#[pyclass]
#[pyo3(name = "TextParser")]
pub(crate) struct WrappedParserTextOnly(Arc<dyn reductionml_core::parsers::TextModeParser>);

#[pyclass]
#[pyo3(name = "JsonParser")]
pub(crate) struct WrappedParserTextAndJson(Arc<dyn reductionml_core::parsers::TextModeParser>);

pub(crate) enum WrappedParser {
    WrappedParserTextOnly(WrappedParserTextOnly),
    WrappedParserTextAndJson(WrappedParserTextAndJson),
}

impl IntoPy<PyObject> for WrappedParser {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            WrappedParser::WrappedParserTextOnly(x) => x.into_py(py),
            WrappedParser::WrappedParserTextAndJson(x) => x.into_py(py),
        }
    }
}

unsafe impl Send for WrappedParserTextOnly {}
unsafe impl Sync for WrappedParserTextOnly {}
unsafe impl Send for WrappedParserTextAndJson {}
unsafe impl Sync for WrappedParserTextAndJson {}

#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormatType {
    VwText,
    Json,
    DsJson,
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

#[pyfunction]
pub(crate) fn create_parser(
    format_type: FormatType,
    features_type: WrappedFeaturesType,
    label_type: WrappedLabelType,
    hash_seed: u32,
    num_bits: u8,
) -> Result<WrappedParser, PyErr> {
    let parser = format_type.get_parser(features_type.into(), label_type.into(), hash_seed, num_bits);
    match format_type {
        FormatType::VwText => Ok(WrappedParser::WrappedParserTextOnly(WrappedParserTextOnly(
            parser.into(),
        ))),
        FormatType::Json => Ok(WrappedParser::WrappedParserTextAndJson(
            WrappedParserTextAndJson(parser.into()),
        )),
        FormatType::DsJson => Ok(WrappedParser::WrappedParserTextAndJson(
            WrappedParserTextAndJson(parser.into()),
        )),
    }
}

#[pymethods]
impl WrappedParserTextOnly {
    /// parse(input: str) -> typing.Tuple[typing.Union[SparseFeatures, CbAdfFeatures], typing.Optional[Union[SimpleLabel, CbLabel]]]
    fn parse(
        &self,
        input: &str,
    ) -> Result<(WrappedFeaturesForReturn, Option<WrappedLabel>), PyErr> {
        let (feats, label) = self
            .0
            .parse_chunk(input)
            .map_err(|x| WrappedError::from(x))?;
        let feats: WrappedFeaturesForReturn =
            feats.try_into().map_err(|x| WrappedError::from(x))?;
        let label: Option<WrappedLabel> = label.map(|x| x.into());
        Ok((feats, label))
    }
}

#[derive(FromPyObject)]
pub(crate) enum JsonInputKinds<'a> {
    StringInput(&'a str),
    DictInput(&'a PyDict),
}

#[pymethods]
impl WrappedParserTextAndJson {
    /// parse(input: typing.Union[typing.Dict[str, typing.Any], str]) -> typing.Tuple[typing.Union[SparseFeatures, CbAdfFeatures], typing.Optional[Union[SimpleLabel, CbLabel]]]
    fn parse(
        &self,
        input: JsonInputKinds,
    ) -> Result<(WrappedFeaturesForReturn, Option<WrappedLabel>), PyErr> {
        let input: Cow<str> = match input {
            JsonInputKinds::StringInput(x) => x.into(),
            JsonInputKinds::DictInput(x) => {
                // TODO: avoid the trip to string here.
                let input: serde_json::Value = depythonize(x).unwrap();
                serde_json::to_string(&input).unwrap().into()
            }
        };

        let (feats, label) = self
            .0
            .parse_chunk(&input)
            .map_err(|x| WrappedError::from(x))?;
        let feats: WrappedFeaturesForReturn =
            feats.try_into().map_err(|x| WrappedError::from(x))?;
        let label: Option<WrappedLabel> = label.map(|x| x.into());
        Ok((feats, label))
    }
}
