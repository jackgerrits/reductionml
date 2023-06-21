use clap::Args;

use std::fs::File;
use std::io::{self, BufRead};

use crate::command::Command;

use anyhow::Result;

#[derive(Args)]
pub(crate) struct CreateInvHashTableArgs {
    // Note: in future, may need the config to handle things like injected features and interactions.
    #[arg(short, long)]
    data: String,

    /// Seed to use when hashing input text
    #[arg(long)]
    #[arg(default_value = "0")]
    hash_seed: u32,

    /// Seed to use when hashing input text
    #[arg(long)]
    #[arg(default_value = "18")]
    num_bits: u8,
}

pub(crate) struct CreateInvHashTableCommand;

impl Command for CreateInvHashTableCommand {
    type Args = CreateInvHashTableArgs;
    fn execute(args: &CreateInvHashTableArgs, _quiet: bool) -> Result<()> {
        let file = File::open(&args.data).unwrap();
        let inv_hash_table = reductionml_core::inverse_hash_table::InverseHashTable::new();

        for _line in io::BufReader::new(file).lines() {
            todo!()
        }

        println!("{}", serde_json::to_string_pretty(&inv_hash_table).unwrap());

        Ok(())
    }
}
