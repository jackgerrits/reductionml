use clap::{Args, Parser, Subcommand, ValueEnum};
use colored::Colorize;

use crate::command::Command;

mod check;
mod command;
mod convert_data;
mod create_inv_hash_table;
mod export_model;
mod gen_completions;
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
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train(train::TrainArgs),
    /// Test a model on a dataset
    Test(test::TestArgs),
    /// Check a config file for errors
    Check(check::CheckArgs),
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
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct InputConfigArg {
    /// Start new model with given configuration
    #[arg(short, long)]
    config: Option<String>,

    /// Load an existing model file
    #[arg(short, long)]
    input_model: Option<String>,
}

fn main() {
    eprintln!("{} This CLI tool is not stable.", "Warning:".yellow());
    let cli = Cli::parse();
    match &cli.command {
        Commands::Check(args) => {
            check::CheckCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::Train(args) => {
            train::TrainCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::CreateInvHashTable(args) => {
            create_inv_hash_table::CreateInvHashTableCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::Test(args) => {
            test::TestCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::ExportModel(args) => {
            export_model::ExportModelCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::ImportModel(args) => {
            import_model::ImportModelCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::ConvertData(args) => {
            convert_data::ConvertDataCommand::execute(args, cli.quiet).unwrap();
        }
        Commands::GenCompletions(args) => {
            gen_completions::GenCompletionsCommand::execute(args, cli.quiet).unwrap();
        }
    }
}
