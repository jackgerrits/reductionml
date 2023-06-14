use clap::Args;
use reductionml_core::reductions::{CBAdfConfig, CoinRegressorConfig};
use reductionml_core::{reduction_registry::REDUCTION_REGISTRY, config_schema::ConfigSchema};

use crate::command::Command;

use anyhow::Result;

#[derive(Args)]
pub(crate) struct GenSchemaArgs {
}

pub(crate) struct GenSchemaCommand;

impl Command for GenSchemaCommand {
    type Args = GenSchemaArgs;
    fn execute(args: &GenSchemaArgs, quiet: bool) -> Result<()> {
        let mut schema = ConfigSchema::new();
    REDUCTION_REGISTRY.read().as_ref().as_ref().unwrap().iter().for_each(|x| {
        schema.add_reduction(x);
    });
    println!("{}", serde_json::to_string_pretty(schema.schema()).unwrap());
    Ok(())
    }
}

