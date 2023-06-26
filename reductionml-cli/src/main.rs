use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum, ValueHint};
use owo_colors::OwoColorize;
use reductionml_core::{
    object_pool::Pool,
    parsers::{TextModeParser, TextModeParserFactory},
    sparse_namespaced_features::SparseFeatures,
    FeaturesType, LabelType,
};

use crate::command::Command;

mod command;
mod config;
mod convert_data;
mod create_inv_hash_table;
mod export_model;
mod gen_completions;
mod gen_schema;
mod import_model;
mod test;
mod train;

#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Silence all output
    #[arg(long, default_value = "false")]
    quiet: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum DataFormat {
    VWText,
    Dsjson,
}

impl DataFormat {
    pub fn get_parser(
        &self,
        features_type: FeaturesType,
        label_type: LabelType,
        hash_seed: u32,
        num_bits: u8,
        pool: Arc<Pool<SparseFeatures>>,
    ) -> Box<dyn TextModeParser> {
        match self {
            DataFormat::VWText => Box::new(
                reductionml_core::parsers::VwTextParserFactory::default().create(
                    features_type,
                    label_type,
                    hash_seed,
                    num_bits,
                    pool,
                ),
            ),
            DataFormat::Dsjson => Box::new(
                reductionml_core::parsers::DsJsonParserFactory::default().create(
                    features_type,
                    label_type,
                    hash_seed,
                    num_bits,
                    pool,
                ),
            ),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train(train::TrainArgs),
    /// Test a model on a dataset
    Test(test::TestArgs),
    /// Check or generate a config
    Config(config::ConfigArgs),
    /// Export a model to a human-readable format
    ExportModel(export_model::ExportModelArgs),
    /// Import a model from a human-readable format
    ImportModel(import_model::ImportModelArgs),
    /// Create an inverse hash table from data
    CreateInvHashTable(create_inv_hash_table::CreateInvHashTableArgs),
    /// Convert between data formats
    ConvertData(convert_data::ConvertDataArgs),
    /// Generate shell completions
    GenCompletions(gen_completions::GenCompletionsArgs),
    /// Generate JSON schema for configuration
    GenSchema(gen_schema::GenSchemaArgs),
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct InputConfigArg {
    /// Start new model with given configuration
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    config: Option<String>,

    /// Load an existing model file
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    input_model: Option<String>,
}

fn handle_args(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Config(args) => {
            config::ConfigCommand::execute(args, cli.quiet)?;
        }
        Commands::Train(args) => {
            train::TrainCommand::execute(args, cli.quiet)?;
        }
        Commands::CreateInvHashTable(args) => {
            create_inv_hash_table::CreateInvHashTableCommand::execute(args, cli.quiet)?;
        }
        Commands::Test(args) => {
            test::TestCommand::execute(args, cli.quiet)?;
        }
        Commands::ExportModel(args) => {
            export_model::ExportModelCommand::execute(args, cli.quiet)?;
        }
        Commands::ImportModel(args) => {
            import_model::ImportModelCommand::execute(args, cli.quiet)?;
        }
        Commands::ConvertData(args) => {
            convert_data::ConvertDataCommand::execute(args, cli.quiet)?;
        }
        Commands::GenCompletions(args) => {
            gen_completions::GenCompletionsCommand::execute(args, cli.quiet)?;
        }
        Commands::GenSchema(args) => {
            gen_schema::GenSchemaCommand::execute(args, cli.quiet)?;
        }
    }
    Ok(())
}

fn main() {
    eprintln!("{}: This CLI tool is not stable", "warning".yellow().bold());
    let cli = Cli::parse();
    if let Err(e) = handle_args(cli) {
        eprintln!("{}: {:?}", "error".red().bold(), e);
        std::process::exit(1);
    }
}
