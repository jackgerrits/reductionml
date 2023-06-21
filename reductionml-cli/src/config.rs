use clap::{Args, ValueEnum};
use reductionml_core::global_config;
use serde_json::json;

use crate::command::Command;

use anyhow::Result;

use clap::Subcommand;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ConfigFormat {
    Json,
    Yaml,
}

#[derive(Args)]
pub(crate) struct ConfigCheckArgs {
    config: String,
}

#[derive(Args)]
pub(crate) struct ConfigNewArgs {
    reduction: String,
    // #[arg(short, long)]
    // format: String,
}

#[derive(Subcommand)]
enum ConfigSubCommand {
    /// Check a model configuration for validity, including reduction type checking.
    Check(ConfigCheckArgs),
    /// Emit a full configuration with all defaults.
    New(ConfigNewArgs),
}

#[derive(Args)]
pub(crate) struct ConfigArgs {
    #[command(subcommand)]
    subcommand: ConfigSubCommand,
}

pub(crate) struct ConfigCommand;

impl Command for ConfigCommand {
    type Args = ConfigArgs;
    fn execute(args: &ConfigArgs, _quiet: bool) -> Result<()> {
        match &args.subcommand {
            ConfigSubCommand::Check(args) => {
                // load json from file
                let json = std::fs::read_to_string(&args.config).unwrap();
                let _workspace =
                    reductionml_core::workspace::Workspace::create_from_json(&json).unwrap();
                println!("ok");
                Ok(())
            }
            ConfigSubCommand::New(args) => {
                let default_reduction_config =
                    reductionml_core::reduction_registry::REDUCTION_REGISTRY
                        .read()
                        .as_ref()
                        .as_ref()
                        .unwrap()
                        .get(&args.reduction)
                        .unwrap()
                        .get_config_default();
                let default_global = global_config::GlobalConfig::default();
                let overall_config = json!({
                    // "$schema": "https://raw.githubusercontent.com/reductionml/reductionml-core/main/config_schema.json",
                    "globalConfig": default_global,
                    "entryReduction": {
                        "typename": args.reduction,
                        "config": default_reduction_config
                    }
                }
                );
                let json = serde_json::to_string_pretty(&overall_config).unwrap();
                println!("{}", json);
                Ok(())
            }
        }
    }
}
