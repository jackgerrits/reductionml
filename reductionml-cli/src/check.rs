use clap::Args;
use reductionml_core::reductions::{CBAdfConfig, CoinRegressorConfig};

use crate::command::Command;

use anyhow::Result;

#[derive(Args)]
pub(crate) struct CheckArgs {
    config: String,
    #[arg(long)]
    show_types: bool,
}

pub(crate) struct CheckCommand;

impl Command for CheckCommand {
    type Args = CheckArgs;
    fn execute(args: &CheckArgs, quiet: bool) -> Result<()> {
        // load json from file
        let json = std::fs::read_to_string(&args.config).unwrap();
        let _workspace = reductionml_core::workspace::Workspace::create_from_json(&json).unwrap();
        println!("ok");
        Ok(())
    }
}
