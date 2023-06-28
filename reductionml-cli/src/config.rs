use std::ops::Deref;

use clap::{Args, ValueEnum};
use reductionml_core::global_config;
use serde_json::json;

use crate::command::Command;

use anyhow::{Context, Result};

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
                let json = std::fs::read_to_string(&args.config).with_context(|| {
                    format!("Failed to read configuration file {}", args.config)
                })?;
                let _workspace = reductionml_core::workspace::Workspace::create_from_json(&json)
                    .with_context(|| {
                        format!("Failed to parse configuration file {}", args.config)
                    })?;
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
                        .with_context(|| {
                            let available_reductions =
                                reductionml_core::reduction_registry::REDUCTION_REGISTRY
                                    .read()
                                    .as_ref()
                                    .as_ref()
                                    .unwrap()
                                    .deref()
                                    .deref()
                                    .iter()
                                    .map(|s| s.typename().to_string())
                                    .collect::<Vec<String>>()
                                    .join(", ");
                            format!(
                                "Reduction \"{}\" does not exist. Available reductions are: {}",
                                args.reduction, available_reductions
                            )
                        })?
                        .get_config_default();
                let default_global = global_config::GlobalConfig::default();
                let overall_config = json!({
                    "$schema": "https://jackgerrits.github.io/reductionml/schema/config_schema.json",
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
