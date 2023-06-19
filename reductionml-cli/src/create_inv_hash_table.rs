use clap::Args;
// use crossterm::{cursor, terminal, ExecutableCommand};

use std::io::{self, BufRead, Write};
use std::{fs::File, io::stderr};

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
    fn execute(args: &CreateInvHashTableArgs, quiet: bool) -> Result<()> {
        let file = File::open(&args.data).unwrap();
        let mut stderr = stderr();
        writeln!(stderr, "Reading data file: {}", &args.data).unwrap();
        writeln!(stderr, "Example count: 0").unwrap();

        let mut counter: i32 = 0;
        let inv_hash_table = reductionml_core::inverse_hash_table::InverseHashTable::new();

        for line in io::BufReader::new(file).lines() {
            counter += 1;
            let _l = line.unwrap();
            // let parsed_line = reductionml::vw_text_parser::extract_features_text(&l).unwrap();
            todo!()

            // if (counter % 1000) == 0 {
            //     stderr.execute(cursor::MoveUp(1)).unwrap();
            //     stderr
            //         .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
            //         .unwrap();
            //     writeln!(stderr, "Example count: {}", counter).unwrap();
            // }

            // let res = parsed_line.iter().flat_map(|(namespace, features)| {
            //     features
            //         .iter()
            //         .map(move |feature| Feature::from_parsed_feature(feature, namespace))
            // });
            // for feature in res {
            //     inv_hash_table.insert(
            //         feature
            //             .hash(args.hash_seed)
            //             .mask(FeatureMask::from_num_bits(args.num_bits)),
            //         feature,
            //     );
            // }
        }

        // stderr.execute(cursor::MoveUp(1)).unwrap();
        // stderr
        //     .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
        //     .unwrap();
        // writeln!(stderr, "Example count: {}", counter).unwrap();
        // println!("{}", serde_json::to_string_pretty(&inv_hash_table).unwrap());

        Ok(())
    }
}
