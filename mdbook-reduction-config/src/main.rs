use clap::{Arg, ArgMatches, Command};
use mdbook::book::{Book, Chapter};
use mdbook::errors::Error;
use mdbook::preprocess::{CmdPreprocessor, Preprocessor, PreprocessorContext};
use mdbook::BookItem;
use pulldown_cmark::{Event, Parser, Tag};
use pulldown_cmark_to_cmark::cmark_resume;
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

fn get_type(prop: &serde_json::Value) -> String {
    match prop {
        serde_json::Value::Bool(_) => "bool".to_string(),
        serde_json::Value::Number(_) => "number".to_string(),
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "array".to_string(),
        serde_json::Value::Object(_) => "reduction".to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn get_default_value(prop: &serde_json::Value) -> String {
    match prop {
        serde_json::Value::Bool(value) => value.to_string(),
        // TODO: fix this for integers.
        serde_json::Value::Number(value) => ((value.as_f64().unwrap() *100.0).round() /100.0).to_string(),
        serde_json::Value::String(value) => value.to_string(),
        serde_json::Value::Array(_) => todo!(),
        // For now we will assume this always corresponds to a reduction
        serde_json::Value::Object(obj) => obj.get("typename").unwrap().as_str().unwrap().to_owned(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn generate_md_events_for_prop<'a>(key: &'a str, prop: &serde_json::Value) -> Vec<Event<'a>> {
    let mut events = vec![];
    events.push(Event::Start(Tag::Item));
    events.push(Event::Code(key.into()));
    events.push(Event::Text("(".into()));
    events.push(Event::Code(get_type(prop).into()));
    events.push(Event::Text("), default = ".into()));
    events.push(Event::Code(get_default_value(prop).into()));
    events.push(Event::End(Tag::Item));
    events
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

                state = cmark_resume(
                    std::iter::once(Event::Start(Tag::List(None))),
                    &mut buf,
                    state.take(),
                )
                .unwrap()
                .into();
                for (key, value) in reduction_config.as_object_mut().unwrap().iter_mut() {
                    let events = generate_md_events_for_prop(key, value);
                    state = cmark_resume(events.into_iter(), &mut buf, state.take())
                        .unwrap()
                        .into();
                }
                state = cmark_resume(
                    std::iter::once(Event::End(Tag::List(None))),
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

    fn run(&self, _ctx: &PreprocessorContext, mut book: Book) -> Result<Book, Error> {
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
