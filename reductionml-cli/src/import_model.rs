use clap::Args;
use colored::Colorize;
use reductionml_core::workspace::Workspace;

use crate::command::Command;

use anyhow::Result;

// TODO: add warning that weights are not printed in any nice way at the moment
#[derive(Args)]
pub(crate) struct ImportModelArgs {
    input_file: String,

    #[arg(short, long)]
    output_model: String,
}

pub(crate) struct ImportModelCommand;

impl Command for ImportModelCommand {
    type Args = ImportModelArgs;
    fn execute(args: &ImportModelArgs, _quiet: bool) -> Result<()> {
        eprintln!(
            "{} Importing a model from JSON is not a supported feature. Use at your own risk.",
            "Warning:".yellow()
        );

        let input_data = std::fs::read(&args.input_file).unwrap();
        let json = serde_json::from_slice(&input_data).unwrap();
        let workspace = Workspace::deserialize_from_json(&json).unwrap();
        let data = workspace.serialize_model().unwrap();
        std::fs::write(&args.output_model, data).unwrap();

        Ok(())
    }
}
