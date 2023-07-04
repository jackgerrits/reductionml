use core::f32;

use derive_more::TryInto;

use smallvec::SmallVec;

use crate::error::{Error, Result};

use crate::object_pool::Pool;
use crate::parsers::ParsedFeature;
use crate::sparse_namespaced_features::{Namespace, SparseFeatures};
use crate::types::{Features, Label, LabelType};
use crate::utils::AsInner;
use crate::{CBAdfFeatures, CBLabel, FeatureMask, FeaturesType, SimpleLabel};

use super::{ParsedNamespaceInfo, TextModeParser, TextModeParserFactory};

#[derive(Clone, Copy)]
struct CBTextLabel {
    shared: bool,
    // Action, cost, prob
    acp: Option<(u32, f32, f32)>,
}

#[derive(TryInto, Clone, Copy)]
enum TextLabel {
    Simple(f32, Option<f32>),
    // Binary(bool),
    CB(CBTextLabel),
}

impl AsInner<CBTextLabel> for TextLabel {
    fn as_inner(&self) -> Option<&CBTextLabel> {
        match self {
            TextLabel::CB(f) => Some(f),
            _ => None,
        }
    }
    fn as_inner_mut(&mut self) -> Option<&mut CBTextLabel> {
        match self {
            TextLabel::CB(f) => Some(f),
            _ => None,
        }
    }
}

// TODO work out where to put tag.
// Idea - tag is not a concept here but for the cases where it was necessary (ccb) it will be folded into the feature type
fn finalize_parsed_result_singleline<'a>(
    parsed: TextParseResult,
    _num_bits: u8,
    dest: SparseFeatures,
) -> (Features<'a>, Option<Label>) {
    let hashed_sparse_features = Features::SparseSimple(dest);
    match parsed.label {
        // TODO fix
        Some(TextLabel::Simple(x, weight)) => (
            hashed_sparse_features,
            Some(Label::Simple(SimpleLabel::new(x, weight.unwrap_or(1.0)))),
        ),
        // TODO binary
        Some(_) => todo!(),
        None => (hashed_sparse_features, None),
    }
}

fn finalize_parsed_result_multiline<'a, 'b, T, U>(
    mut feats_iter: T,
    parsed: U,
    expected_label: LabelType,
    expected_features: FeaturesType,
    _num_bits: u8,
) -> Result<(Features<'b>, Option<Label>)>
where
    T: IntoIterator<Item = SparseFeatures> + Iterator<Item = SparseFeatures> + Clone,
    U: Iterator<Item = TextParseResult<'a>>,
{
    match (expected_label, expected_features) {
        (LabelType::CB, FeaturesType::SparseCBAdf) => {
            // First thing to do is to determine if there is a shared example.
            let mut txt_labels_iter = parsed.map(|x| x.label.unwrap()).peekable();
            // let mut feats_iter = feats.into_iter();
            let first_label: &CBTextLabel = txt_labels_iter
                .peek()
                .ok_or(Error::InvalidArgument("".to_owned()))?
                .as_inner()
                .expect("Label should be CB");
            let first_is_shared = first_label.shared;

            // TODO assert not more than 1 is shared.
            let shared_ex = if first_is_shared {
                // Consume shared token
                txt_labels_iter.next();
                Some(feats_iter.next().unwrap())
            } else {
                None
            };

            // Find the labelled action.
            let mut label: Option<CBLabel> = None;
            for (counter, action_label) in txt_labels_iter.enumerate() {
                let lbl: &CBTextLabel = action_label.as_inner().expect("Label should be CB");
                if let Some((_a, c, p)) = lbl.acp {
                    if label.is_some() {
                        return Err(Error::InvalidArgument(
                            "More than one action label found".to_owned(),
                        ));
                    }
                    label = Some(CBLabel {
                        action: counter,
                        cost: c,
                        probability: p,
                    });
                }
            }

            Ok((
                Features::SparseCBAdf(CBAdfFeatures {
                    shared: shared_ex,
                    actions: feats_iter.collect(),
                }),
                label.map(Label::CB),
            ))
        }
        _ => Err(Error::InvalidArgument("".to_owned())),
    }
}

struct TextParseResult<'a> {
    _tag: Option<&'a str>,
    // namespaces: Vec<ParsedNamespace<'a>>,
    label: Option<TextLabel>,
}

fn parse_label(tokens: &[&str], label_type: LabelType) -> Result<Option<TextLabel>> {
    match label_type {
        LabelType::Simple => match tokens.len() {
            0 => Ok(None),
            1 => Ok(Some(TextLabel::Simple(
                fast_float::parse(tokens[0]).unwrap(),
                None,
            ))),
            2 => Ok(Some(TextLabel::Simple(
                fast_float::parse(tokens[0]).unwrap(),
                Some(fast_float::parse(tokens[1]).unwrap()),
            ))),
            // Initial not currently supported...
            3 => todo!(),
            _ => todo!(),
        },
        LabelType::Binary => todo!(),
        LabelType::CB => match tokens.iter().next() {
            None => Ok(Some(TextLabel::CB(CBTextLabel {
                shared: false,
                acp: None,
            }))),
            Some(value) if value.trim() == "shared" => Ok(Some(TextLabel::CB(CBTextLabel {
                shared: true,
                acp: None,
            }))),
            Some(value) => {
                let mut tokens = value.split(':');
                let action = tokens.next().unwrap().parse().unwrap();
                let cost = fast_float::parse(tokens.next().unwrap()).unwrap();
                let probability = fast_float::parse(tokens.next().unwrap()).unwrap();

                // TODO: check that there are no more tokens

                Ok(Some(TextLabel::CB(CBTextLabel {
                    shared: false,
                    acp: Some((action, cost, probability)),
                })))
            }
        },
    }
}

// TODO - consider conditionally allowing a feature whose name is a number ONLY to be interpreted as an anonymous features
// This would be to mimic VW's hash "mode" of all vs txt
fn parse_feature<'a>(feature: &'a str, offset_counter: &mut u32) -> (ParsedFeature<'a>, f32) {
    // Check if char 0 is a :
    let first_char_is_colon = feature.starts_with(':');
    if first_char_is_colon {
        // Anonymous feature
        let value = fast_float::parse(&feature[1..]);
        if let Ok(value) = value {
            let offset_to_use = *offset_counter;
            *offset_counter += 1;
            (
                ParsedFeature::Anonymous {
                    offset: offset_to_use,
                },
                value,
            )
        } else {
            return (
                ParsedFeature::SimpleWithStringValue {
                    name: "",
                    value: feature[1..].trim(),
                },
                1.0,
            );
        }
    } else {
        // Named feature
        let mut tokens = feature.split(':');
        let name = tokens.next().unwrap();
        match tokens.next() {
            Some(value) => {
                if let Ok(value) = fast_float::parse(value) {
                    (ParsedFeature::Simple { name }, value)
                } else {
                    (
                        ParsedFeature::SimpleWithStringValue {
                            name,
                            value: value.trim(),
                        },
                        1.0,
                    )
                }
            }
            None => (ParsedFeature::Simple { name }, 1.0),
        }
    }
}

fn parse_namespace_inline(
    namespace_segment: &str,
    dest_namespace: &mut SparseFeatures,
    hash_seed: u32,
    num_bits: u8,
) -> Result<()> {
    // Check if first char is a space or not
    let first_char_is_space = namespace_segment.starts_with(' ');
    let mut tokens = namespace_segment.split_ascii_whitespace();

    let (namespace_name, namespace_value) = if first_char_is_space {
        // Anonymous namespace
        (" ", 1.0)
    } else {
        let namespace_info_token = tokens.next().unwrap();
        let mut namespace_info_tokens = namespace_info_token.split(':');
        let name = namespace_info_tokens.next().unwrap();
        let value = match namespace_info_tokens.next() {
            Some(value) => fast_float::parse(value).unwrap(),
            None => 1.0,
        };

        (name, value)
    };

    let namespace_def = Namespace::from_name(namespace_name, hash_seed);
    let namespace_hash = namespace_def.hash(hash_seed);

    let dest = dest_namespace.get_or_create_namespace(namespace_def);
    let mut offset_counter = 0;
    for token in tokens {
        let (parsed_feat, feat_value) = parse_feature(token, &mut offset_counter);
        // let this_ns = dest.get_or_create_namespace_with_capacity(namespace_hash, features.len());
        let feature_hash = parsed_feat.hash(namespace_hash);
        let masked_hash = feature_hash.mask(FeatureMask::from_num_bits(num_bits));
        dest.add_feature(masked_hash, feat_value * namespace_value);
    }

    Ok(())
}

fn parse_namespace_info_token(namespace_segment: &str) -> Result<(&str, f32)> {
    let mut tokens: std::str::Split<char> = namespace_segment.split(':');
    let name = tokens
        .next()
        .ok_or(Error::ParserError("Expected namespace name".to_owned()))?;
    let value = match tokens.next() {
        Some(value) => fast_float::parse(value).map_err(|err| {
            Error::ParserError(format!("Failed to parse namespace value: {}", err))
        })?,
        None => 1.0,
    };

    Ok((name, value))
}

// "Consumes" some amount of input and returns the namespace info and the remaining input
fn parse_namespace_info(input: &str) -> Result<(&str, (ParsedNamespaceInfo, f32))> {
    let first_char_is_space = input.starts_with(' ');
    // Extract up until the first space
    if first_char_is_space {
        Ok((&input[1..], (ParsedNamespaceInfo::Default, 1.0)))
    } else {
        let input_until_first_space = input.find(' ').unwrap();
        let namespace_info_token = &input[..input_until_first_space];
        let (ns_name, ns_value) = parse_namespace_info_token(namespace_info_token)?;
        Ok((
            &input[input_until_first_space..],
            (ParsedNamespaceInfo::Named(ns_name), ns_value),
        ))
    }
}

fn extract_namespace_features(
    namespace_segment: &str,
) -> Result<(ParsedNamespaceInfo, Vec<ParsedFeature>)> {
    let (remaining, (namespace_name, _namespace_value)) = parse_namespace_info(namespace_segment)?;

    let tokens = remaining.split_ascii_whitespace();
    let mut offset_counter = 0;
    let extracted_featrues = tokens
        .map(|x| {
            let (feat, _value) = parse_feature(x, &mut offset_counter);
            feat
        })
        .collect();
    Ok((namespace_name, extracted_featrues))
}

// TODO revisit this function. Scanning to the last character is not ideal since it is linear time.
fn parse_initial_segment(
    text: &str,
    label_type: LabelType,
) -> Result<(Option<&str>, Option<TextLabel>)> {
    // Is the last char of text a space?
    let last_char_is_space = text.ends_with(' ');

    // TODO: avoid this allocation!
    let mut tokens: Vec<&str> = text.split_whitespace().collect();

    let tag = match tokens.last() {
        Some(&x) if (x.starts_with('\'') || !last_char_is_space) => {
            tokens.pop();
            if let Some(x) = x.strip_prefix('\'') {
                Some(x)
            } else {
                Some(x)
            }
        }
        Some(_) => None,
        None => None,
    };

    let label = parse_label(&tokens, label_type)?;
    Ok((tag, label))
}

fn parse_text_line_internal<'a>(
    text: &'a str,
    label_type: LabelType,
    dest: &mut SparseFeatures,
    hash_seed: u32,
    num_bits: u8,
) -> Result<TextParseResult<'a>> {
    // Get string view up until first bar
    let mut segments = text.split('|');
    let initial_segment = segments.next().unwrap();
    let (tag, label) = parse_initial_segment(initial_segment, label_type)?;

    for segment in segments {
        parse_namespace_inline(segment, dest, hash_seed, num_bits)?;
    }
    Ok(TextParseResult { _tag: tag, label })
}

#[derive(Default)]
pub struct VwTextParserFactory;
impl TextModeParserFactory for VwTextParserFactory {
    type Parser = VwTextParser;
    fn create(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
        pool: std::sync::Arc<Pool<SparseFeatures>>,
    ) -> Self::Parser {
        VwTextParser {
            feature_type: features_type,
            label_type,
            hash_seed,
            num_bits,
            pool,
        }
    }
}

pub struct VwTextParser {
    feature_type: FeaturesType,
    label_type: LabelType,
    hash_seed: u32,
    num_bits: u8,
    pool: std::sync::Arc<Pool<SparseFeatures>>,
}

fn read_multi_lines(
    input: &mut dyn std::io::BufRead,
    mut output_buffer: String,
) -> Result<Option<String>> {
    assert!(output_buffer.is_empty());
    loop {
        let len_before = output_buffer.len();
        if !output_buffer.is_empty() {
            output_buffer.push('\n');
        }
        let bytes_read = input.read_line(&mut output_buffer)?;
        if bytes_read == 0 && output_buffer.is_empty() {
            return Ok(None);
        }
        output_buffer.truncate(output_buffer.trim_end().len());

        // If we encounter an empty line, we are done. But if we are at
        // the start (no data yet) we should just skip the empty line.
        if output_buffer.is_empty() && len_before == 0 {
            continue;
        }

        if len_before > 0 && output_buffer.len() == len_before {
            // We read a line, but it was empty. This means we are done.
            return Ok(Some(output_buffer));
        }
    }
}

fn read_single_line(
    input: &mut dyn std::io::BufRead,
    mut output_buffer: String,
) -> Result<Option<String>> {
    loop {
        let bytes_read = input.read_line(&mut output_buffer)?;
        if bytes_read == 0 {
            return Ok(None);
        }
        output_buffer.truncate(output_buffer.trim_end().len());

        // If we encounter an empty line, we are done. But if we are at
        // the start (no data yet) we should just skip the empty line.
        if output_buffer.is_empty() {
            continue;
        }

        return Ok(Some(output_buffer));
    }
}

impl TextModeParser for VwTextParser {
    fn get_next_chunk(
        &self,
        input: &mut dyn std::io::BufRead,
        mut output_buffer: String,
    ) -> Result<Option<String>> {
        output_buffer.clear();
        if self.is_multiline() {
            read_multi_lines(input, output_buffer)
        } else {
            read_single_line(input, output_buffer)
        }
    }

    fn parse_chunk<'a, 'b>(&self, chunk: &'a str) -> Result<(Features<'b>, Option<Label>)> {
        if self.is_multiline() {
            let mut results = SmallVec::<[TextParseResult<'a>; 4]>::new();
            let mut all_feautures = SmallVec::<[SparseFeatures; 4]>::new();
            for line in chunk.lines() {
                let mut dest = self.pool.get_object();
                let result = parse_text_line_internal(
                    line,
                    self.label_type,
                    &mut dest,
                    self.hash_seed,
                    self.num_bits,
                )?;
                results.push(result);
                all_feautures.push(dest);
            }
            finalize_parsed_result_multiline(
                all_feautures.into_iter(),
                results.into_iter(),
                self.label_type,
                self.feature_type,
                self.num_bits,
            )
        } else {
            let mut dest = self.pool.get_object();
            let result = parse_text_line_internal(
                chunk,
                self.label_type,
                &mut dest,
                self.hash_seed,
                self.num_bits,
            )?;
            Ok(finalize_parsed_result_singleline(
                result,
                self.num_bits,
                dest,
            ))
        }
    }

    fn extract_feature_names<'a>(
        &self,
        chunk: &'a str,
    ) -> Result<std::collections::HashMap<ParsedNamespaceInfo<'a>, Vec<ParsedFeature<'a>>>> {
        if self.is_multiline() {
            chunk
                .lines()
                .flat_map(|line| {
                    let mut segments = line.split('|');
                    let _label_section = segments.next().unwrap();
                    segments.map(extract_namespace_features)
                })
                .collect()
        } else {
            let mut segments = chunk.split('|');
            let _label_section = segments.next().unwrap();
            segments.map(extract_namespace_features).collect()
        }
    }
}

impl VwTextParser {
    fn is_multiline(&self) -> bool {
        self.feature_type == FeaturesType::SparseCBAdf && self.label_type == LabelType::CB
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        error::Error,
        parsers::vw_text_parser::{read_multi_lines, read_single_line},
    };
    use std::io::Cursor;

    #[test]
    fn chunk_multiline() -> Result<(), Error> {
        let input = r#"line 1
line 2"#;

        let mut input = Cursor::new(input);
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, Some("line 1\nline 2".to_string()));
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1
line 2"#;

        let mut input = Cursor::new(input);
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, Some("line 1\nline 2".to_string()));
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1
line 2

        "#;
        let mut input = Cursor::new(input);
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, Some("line 1\nline 2".to_string()));
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1
line 2


line 3
line 4

        "#;
        let mut input = Cursor::new(input);
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, Some("line 1\nline 2".to_string()));
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, Some("line 3\nline 4".to_string()));
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, None);
        let res = read_multi_lines(&mut input, String::new())?;
        assert_eq!(res, None);
        Ok(())
    }

    #[test]
    fn chunk_singleline() -> Result<(), Error> {
        let input = r#"line 1
line 2"#;

        let mut input = Cursor::new(input);
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 1".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 2".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1
line 2"#;

        let mut input = Cursor::new(input);
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 1".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 2".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1

line 2

        "#;
        let mut input = Cursor::new(input);
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 1".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 2".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, None);

        let input = r#"


line 1
line 2


line 3
line 4

        "#;
        let mut input = Cursor::new(input);
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 1".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 2".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 3".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, Some("line 4".to_string()));
        let res = read_single_line(&mut input, String::new())?;
        assert_eq!(res, None);
        Ok(())
    }
}
