use anyhow::Result;

pub(crate) trait Command {
    type Args;
    fn execute(args: &Self::Args) -> Result<()>;
}
