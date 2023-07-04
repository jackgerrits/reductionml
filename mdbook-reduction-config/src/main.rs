use clap::{Arg, ArgMatches, Command};
use core::panic;
use mdbook::book::{Book, Chapter};
use mdbook::errors::Error;
use mdbook::preprocess::{CmdPreprocessor, Preprocessor, PreprocessorContext};
use mdbook::BookItem;
use pulldown_cmark::{CodeBlockKind, CowStr, Event, Parser, Tag};
use pulldown_cmark_to_cmark::{cmark, cmark_resume};
use reductionml_core::reduction_registry::REDUCTION_REGISTRY;
use semver::{Version, VersionReq};
use std::io;
use std::process;

pub fn make_app() -> Command {
    Command::new("preprocessor")
        .about("A mdbook preprocessor which does precisely nothing")
        .subcommand(
            Command::new("supports")
                .arg(Arg::new("renderer").required(true))
                .about("Check whether a renderer is supported by this preprocessor"),
        )
}

fn rewrite_reduction_config(chapter: &mut Chapter) {
    let mut buf = String::with_capacity(chapter.content.len());

    let mut state = None;
    for event in Parser::new(&chapter.content) {
        match event {
            Event::Code(contents) if contents.starts_with("reduction-config-json=") => {
                let reduction_name = contents["reduction-config-json=".len()..].trim();
                let reduction_name = reduction_name.trim_matches('"');
                let mut reduction_config = REDUCTION_REGISTRY
                    .read()
                    .as_ref()
                    .unwrap()
                    .get(reduction_name)
                    .unwrap()
                    .get_config_default();
                for (_, value) in reduction_config.as_object_mut().unwrap().iter_mut() {
                    *value = match value {
                        serde_json::Value::Bool(value) => {
                            serde_json::Value::String(format!("bool, default={}", value))
                        }
                        serde_json::Value::Number(value) => {
                            serde_json::Value::String(format!("number, default={}", value))
                        }
                        serde_json::Value::String(value) => {
                            serde_json::Value::String(format!("string, default={}", value))
                        }
                        serde_json::Value::Array(value) => {
                            serde_json::Value::String("array".to_owned())
                        }
                        serde_json::Value::Object(value) => {
                            serde_json::Value::String("object".to_owned())
                        }
                        _ => panic!("not supported"),
                    }
                }
                let reduction_config = serde_json::to_string_pretty(&reduction_config).unwrap();
                state = cmark_resume(
                    std::iter::once(Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        CowStr::Borrowed("json"),
                    )))),
                    &mut buf,
                    state.take(),
                )
                .unwrap()
                .into();
                state = cmark_resume(
                    std::iter::once(Event::Text(CowStr::Borrowed(&reduction_config))),
                    &mut buf,
                    state.take(),
                )
                .unwrap()
                .into();
                state = cmark_resume(
                    std::iter::once(Event::End(Tag::CodeBlock(CodeBlockKind::Fenced(
                        CowStr::Borrowed("json"),
                    )))),
                    &mut buf,
                    state.take(),
                )
                .unwrap()
                .into();
            }
            a => {
                state = cmark_resume(std::iter::once(a), &mut buf, state.take())
                    .unwrap()
                    .into()
            }
        }
    }
    if let Some(state) = state {
        state.finalize(&mut buf).unwrap();
    }
    chapter.content = buf;
}

pub struct ReductionConfigPreprocessor;

impl ReductionConfigPreprocessor {
    pub fn new() -> ReductionConfigPreprocessor {
        ReductionConfigPreprocessor
    }
}

impl Preprocessor for ReductionConfigPreprocessor {
    fn name(&self) -> &str {
        "reduction-config-preprocessor"
    }

    fn run(&self, ctx: &PreprocessorContext, mut book: Book) -> Result<Book, Error> {
        book.for_each_mut(|x| {
            if let BookItem::Chapter(chapter) = x {
                rewrite_reduction_config(chapter);
            }
        });

        Ok(book)
    }

    fn supports_renderer(&self, renderer: &str) -> bool {
        renderer != "not-supported"
    }
}

fn main() {
    let matches = make_app().get_matches();

    // Users will want to construct their own preprocessor here
    let preprocessor = ReductionConfigPreprocessor::new();

    if let Some(sub_args) = matches.subcommand_matches("supports") {
        handle_supports(&preprocessor, sub_args);
    } else if let Err(e) = handle_preprocessing(&preprocessor) {
        eprintln!("{}", e);
        process::exit(1);
    }
}

fn handle_preprocessing(pre: &dyn Preprocessor) -> Result<(), Error> {
    let (ctx, book) = CmdPreprocessor::parse_input(io::stdin())?;

    let book_version = Version::parse(&ctx.mdbook_version)?;
    let version_req = VersionReq::parse(mdbook::MDBOOK_VERSION)?;

    if !version_req.matches(&book_version) {
        eprintln!(
            "Warning: The {} plugin was built against version {} of mdbook, \
             but we're being called from version {}",
            pre.name(),
            mdbook::MDBOOK_VERSION,
            ctx.mdbook_version
        );
    }

    let processed_book = pre.run(&ctx, book)?;
    serde_json::to_writer(io::stdout(), &processed_book)?;

    Ok(())
}

fn handle_supports(pre: &dyn Preprocessor, sub_args: &ArgMatches) -> ! {
    let renderer = sub_args
        .get_one::<String>("renderer")
        .expect("Required argument");
    let supported = pre.supports_renderer(renderer);

    // Signal whether the renderer is supported by exiting with 1 or 0.
    if supported {
        process::exit(0);
    } else {
        process::exit(1);
    }
}
