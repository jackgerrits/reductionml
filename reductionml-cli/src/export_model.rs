use clap::Args;
use colored::Colorize;
use reductionml_core::workspace::{self, Workspace};

use crate::command::Command;

use anyhow::Result;

// TODO: add warning that weights are not printed in any nice way at the moment
#[derive(Args)]
pub(crate) struct ExportModelArgs {
    #[arg(short, long)]
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
        workspace.serialize_to_json().unwrap();
        let json = serde_json::to_string_pretty(&workspace).unwrap();
        println!("{}", json);
        Ok(())

        // todo!()
    }
}
