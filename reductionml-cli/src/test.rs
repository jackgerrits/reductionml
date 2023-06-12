use clap::Args;
use colored::Colorize;

use crate::{command::Command, DataFormat};

use anyhow::Result;

#[derive(Args)]
pub(crate) struct TestArgs {
    #[arg(short, long)]
    data: String,

    #[arg(long)]
    #[arg(default_value = "vw-text")]
    data_format: DataFormat,

    /// Load an existing model file
    #[arg(short, long)]
    input_model: Option<String>,

    /// Seed to use when hashing input text
    #[arg(long)]
    #[arg(default_value = "0")]
    hash_seed: u32,

    // Output predictions to file
    #[arg(short, long)]
    predictions: Option<String>,

    // Metric values to calculate during training
    #[arg(short, long)]
    #[arg(default_value = "auto")]
    metrics: Option<Vec<String>>,
}

pub(crate) struct TestCommand;

impl Command for TestCommand {
    type Args = TestArgs;
    fn execute(_args: &TestArgs, quiet: bool) -> Result<()> {
        eprintln!("{} Not implemented.", "Error:".red());
        todo!()
    }
}
