use crate::{command::Command, Cli};
use clap::{Args, CommandFactory, ValueEnum};
use clap_complete::{generate, Shell};
use owo_colors::OwoColorize;

use anyhow::Result;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ShellWrapper {
    Bash,
    Fish,
}

impl From<ShellWrapper> for Shell {
    fn from(s: ShellWrapper) -> Self {
        match s {
            ShellWrapper::Bash => Shell::Bash,
            ShellWrapper::Fish => Shell::Fish,
        }
    }
}

#[derive(Args)]
pub(crate) struct GenCompletionsArgs {
    shell: Option<ShellWrapper>,
}

pub(crate) struct GenCompletionsCommand;

impl Command for GenCompletionsCommand {
    type Args = GenCompletionsArgs;
    fn execute(args: &GenCompletionsArgs, _quiet: bool) -> Result<()> {
        match args.shell {
            Some(shell) => {
                generate(
                    Shell::from(shell),
                    &mut Cli::command(),
                    "reductionml-cli",
                    &mut std::io::stdout(),
                );

                eprintln!();
                match shell {
                    ShellWrapper::Bash => eprintln!("{}: To install these completions, run: reductionml-cli gen-completions bash > /usr/share/bash-completion/completions/reductionml-cli", "Hint".blue()),
                    ShellWrapper::Fish => eprintln!("{}: To install these completions, run: reductionml-cli gen-completions fish > ~/.config/fish/completions/reductionml-cli.fish", "Hint".blue()),
                }
            }
            None => {
                eprintln!("Generate shell completions. To install them for your shell, run:");
                eprintln!();
                eprintln!("{}", "Bash:".blue());
                eprintln!("  reductionml-cli gen-completions bash > /usr/share/bash-completion/completions/reductionml-cli");
                eprintln!();
                eprintln!("{}", "Fish:".blue());
                eprintln!("  reductionml-cli gen-completions fish > ~/.config/fish/completions/reductionml-cli.fish");
                eprintln!();
            }
        }
        Ok(())
    }
}
