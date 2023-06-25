use clap::Args;
use owo_colors::OwoColorize;
use reductionml_core::workspace::Workspace;

use crate::command::Command;

use anyhow::Result;

// TODO: add warning that weights are not printed in any nice way at the moment
#[derive(Args)]
pub(crate) struct ExportModelArgs {
    input_model: String,
    // #[arg(short, long)]
    // output_file: String,
}

pub(crate) struct ExportModelCommand;

impl Command for ExportModelCommand {
    type Args = ExportModelArgs;
    fn execute(args: &ExportModelArgs, _quiet: bool) -> Result<()> {
        eprintln!(
            "{} Exporting a model to JSON is not a supported feature. Use at your own risk.",
            "Warning:".yellow()
        );
        let input_model_data = std::fs::read(&args.input_model).unwrap();
        let workspace = Workspace::create_from_model(&input_model_data).unwrap();
        let json = workspace.serialize_to_json().unwrap();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
        Ok(())
    }
}
