use clap::Args;
use colored::Colorize;

use crate::command::Command;

use anyhow::Result;

// TODO: add warning that weights are not printed in any nice way at the moment
#[derive(Args)]
pub(crate) struct ImportModelArgs {
    #[arg(short, long)]
    input_file: String,
    #[arg(short, long)]
    output_model: String,
}

pub(crate) struct ImportModelCommand;

impl Command for ImportModelCommand {
    type Args = ImportModelArgs;
    fn execute(_args: &ImportModelArgs, quiet: bool) -> Result<()> {
        eprintln!(
            "{} Importing a model from JSON is not a supported feature. Use at your own risk.",
            "Warning:".yellow()
        );
        todo!()
    }
}
