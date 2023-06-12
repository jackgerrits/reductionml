use std::{
    fs::File,
    io::{self, stdout, Write}, time::Duration,
};

use anyhow::Result;

use clap::Args;
use crossterm::{cursor, terminal, ExecutableCommand};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use reductionml::{
    metrics::MeanSquaredErrorMetric,
    metrics::Metric,
    object_pool::PoolReturnable,
    parsers::{TextModeParserFactory, VwTextParserFactory},
};

use crate::{command::Command, DataFormat, InputConfigArg};

// TODO: multipass
// TODO: test file for metrics
#[derive(Args)]
pub(crate) struct TrainArgs {
    #[arg(short, long)]
    data: String,

    #[arg(long)]
    #[arg(default_value = "vw-text")]
    data_format: DataFormat,

    #[command(flatten)]
    input_config: InputConfigArg,

    // Save final model to file
    #[arg(short, long)]
    output_model: Option<String>,

    /// Seed to use when hashing input text
    #[arg(long)]
    #[arg(default_value = "0")]
    hash_seed: u32,

    // Output predictions to file
    #[arg(short, long)]
    predictions: Option<String>,

    // Metric values to calculate during training
    #[arg(short, long)]
    #[arg(default_value = "auto")]
    metrics: Option<Vec<String>>,
}

pub(crate) struct TrainCommand;

impl Command for TrainCommand {
    type Args = TrainArgs;
    fn execute(args: &TrainArgs, quiet: bool) -> Result<()> {
        let mut workspace = match (&args.input_config.config, &args.input_config.input_model) {
            // Loading from json config
            (Some(config_file), None) => {
                let json = std::fs::read_to_string(config_file).unwrap();
                reductionml::workspace::Workspace::create_from_json(&json).unwrap()
            }
            // Loading from model file
            (None, Some(input_model_file)) => {
                let data = std::fs::read(input_model_file).unwrap();
                reductionml::workspace::Workspace::create_from_model(&data).unwrap()
            }
            _ => unreachable!(),
        };

        let file = File::open(&args.data).unwrap();
        let mut metric = MeanSquaredErrorMetric::new();

        println!("Reading data file: {}", &args.data);
        println!(
            "Using entry reduction: {}",
            workspace.get_entry_reduction().typename()
        );
        println!();
        println!("Training...");

        // let mut stdout = stdout();
        // writeln!(stdout, "Example count: 0").unwrap();
        // writeln!(stdout, "Metric: 0").unwrap();

        let mut counter: i32 = 0;
        let pool = workspace.features_pool().clone();
        // TODO read format
        let parser_factory = VwTextParserFactory::default();
        let parser = parser_factory.create(
            workspace
                .get_entry_reduction()
                .types()
                .input_features_type(),
            workspace.get_entry_reduction().types().input_label_type(),
            args.hash_seed,
            workspace.global_config().num_bits(),
            pool.clone(),
        );

        let mp = MultiProgress::new();
        let pb: ProgressBar = ProgressBar::new(file.metadata()?.len());
        let ex_spinner = ProgressBar::new_spinner();
        mp.add(ex_spinner.clone());
        mp.add(pb.clone());
        pb.set_style(ProgressStyle::with_template("Input progress: {bar:40.cyan/blue} {bytes_per_sec}")
        .unwrap()
        .progress_chars("##-"));

    // ex_spinner.set_style(ProgressStyle::default_spinner());
    ex_spinner.set_style(ProgressStyle::with_template("Elapsed time: {elapsed}\nExamples processed: {pos}\nPer sec: {per_sec}\n\n").unwrap());
    ex_spinner.set_message(format!("{}", counter));
    ex_spinner.enable_steady_tick(Duration::from_millis(250));
    pb.enable_steady_tick(Duration::from_millis(250));

        let mut f = pb.wrap_read(file);
        let mut input_file = io::BufReader::new(f);
        let mut buffer: String = String::new();
        loop {
            let chunk = parser.get_next_chunk(&mut input_file, buffer).unwrap();
            match chunk {
                Some(data) => {
                    counter += 1;
                    let (features, label) = parser.parse_chunk(&data).unwrap();
                    let prediction = workspace.predict(&features);
                    let label = label.unwrap();
                    // metric.add_point(&label, &prediction);
                    workspace.learn(&features, &label);

                    if (counter % 1) == 0 {
                        // stdout.execute(cursor::MoveUp(2)).unwrap();
                        // stdout
                        //     .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
                        //     .unwrap();

                        // ex_spinner.set_message(format!("{}", counter));

                        // pb.set_message(format!("{}", counter));
                        // writeln!(stdout, "Example count: {}", counter).unwrap();
                        // writeln!(stdout, "Mean squared error: {}", metric.get_value()).unwrap();
                        // mp.println(format!("Mean squared error: {}", metric.get_value()));
                        // pb.tick();
                        // ex_spinner.tick();
                        ex_spinner.inc(1);
                    }

                    // Put feature objects back into the pool for reuse.
                    features.return_object(pool.as_ref());

                    // Transfer ownership so we can reuse the buffer
                    buffer = data;
                }
                None => break,
            }
        }

        pb.finish();
        ex_spinner.finish();
        // pb.println(format!("Mean squared error: {}", metric.get_value()));
        // mp.finish();

        // stdout.execute(cursor::MoveUp(2)).unwrap();
        // stdout
        //     .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
        //     .unwrap();
        // writeln!(stdout, "Example count: {}", counter).unwrap();
        // writeln!(stdout, "Mean squared error: {}", metric.get_value()).unwrap();

        Ok(())
    }
}
