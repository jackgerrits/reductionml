use clap::Args;

use reductionml_core::{config_schema::ConfigSchema, reduction_registry::REDUCTION_REGISTRY};

use crate::command::Command;

use anyhow::Result;

#[derive(Args)]
pub(crate) struct GenSchemaArgs {}

pub(crate) struct GenSchemaCommand;

fn build_json_schema() -> Result<String> {
    let mut schema = ConfigSchema::new();
    REDUCTION_REGISTRY
        .read()
        .as_ref()
        .as_ref()
        .unwrap()
        .iter()
        .for_each(|x| {
            schema.add_reduction(x);
        });
    Ok(serde_json::to_string_pretty(schema.schema()).unwrap())
}

impl Command for GenSchemaCommand {
    type Args = GenSchemaArgs;
    fn execute(_args: &GenSchemaArgs, _quiet: bool) -> Result<()> {
        println!("{}", build_json_schema()?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use valico::json_schema;

    use super::build_json_schema;

    // Not sure if this will actually catch anything, but it's better than nothing
    #[test]
    fn validate_generated_schema() {
        let json_schema: Value = serde_json::from_str(&build_json_schema().unwrap()).unwrap();
        let mut scope = json_schema::Scope::new();
        let _ = scope.compile_and_return(json_schema.clone(), true).unwrap();
    }
}
