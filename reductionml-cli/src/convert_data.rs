use clap::{Args, ValueEnum};

use reductionml_core::LabelType;

use crate::{command::Command, DataFormat};

use anyhow::{anyhow, Result};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum LabelTypeWrapper {
    Simple,
    Binary,
    CB,
}

impl From<LabelTypeWrapper> for LabelType {
    fn from(l: LabelTypeWrapper) -> Self {
        match l {
            LabelTypeWrapper::Simple => LabelType::Simple,
            LabelTypeWrapper::Binary => LabelType::Binary,
            LabelTypeWrapper::CB => LabelType::CB,
        }
    }
}

#[derive(Args)]
pub(crate) struct ConvertDataArgs {
    #[arg(long)]
    from_file: String,

    #[arg(long)]
    from_format: DataFormat,

    // TODO: consider if features type needs to be specified too
    #[arg(long)]
    label_type: LabelTypeWrapper,

    #[arg(long)]
    to_file: String,

    #[arg(long)]
    to_format: DataFormat,
}

pub(crate) struct ConvertDataCommand;

impl Command for ConvertDataCommand {
    type Args = ConvertDataArgs;
    fn execute(_args: &ConvertDataArgs, _quiet: bool) -> Result<()> {
        Err(anyhow!("Not implemented"))
    }
}
