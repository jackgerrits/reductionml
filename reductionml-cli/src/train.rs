use std::{
    fs::File,
    io::{self, stdout, Write},
    time::Duration,
};

use anyhow::Result;

use clap::{Args, ValueHint};
use colored::Colorize;
// use crossterm::{cursor, terminal, ExecutableCommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use prettytable::{cell, table};
use prettytable::{format, Table};
use reductionml_core::{
    metrics::MeanSquaredErrorMetric,
    metrics::{Metric, MetricValue},
    object_pool::{self, PoolReturnable},
    parsers::{TextModeParserFactory, VwTextParserFactory},
    Features, Label,
};

use crate::{command::Command, DataFormat, InputConfigArg};
use rayon::{prelude::*, vec};
use std::str::FromStr;
// TODO: multipass
// TODO: test file for metrics
#[derive(Args)]
pub(crate) struct TrainArgs {
    #[arg(short, long, value_hint = ValueHint::FilePath)]
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
    // TODO: implement this
    #[arg(short, long)]
    predictions: Option<String>,

    // Metric values to calculate during training
    #[arg(short, long)]
    #[arg(default_value = None, value_parser, num_args = 1.., value_delimiter = ',')]
    metrics: Option<Vec<String>>,

    #[arg(long)]
    #[arg(default_value = "128")]
    read_batch_size: usize,

    #[arg(long)]
    #[arg(default_value = "512")]
    queue_size: usize,

    /// Number of threads to use for the rayon thread pool. By default will use
    /// the number of logical cores.
    #[arg(long)]
    #[arg(default_value = None)]
    thread_pool_size: Option<usize>,
}

enum OutputPeriod {
    Additive(i32),
    Multiplicative(f32),
}

struct TrainResultManager {
    iteration: i32,
    next_output_iteration: i32,
    period: OutputPeriod,
    last_render_height: u16,
    table: Table,
    columns: Vec<String>,
}

impl TrainResultManager {
    fn new(period: OutputPeriod, columns: Vec<String>) -> TrainResultManager {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BORDERS_ONLY);
        table.set_titles(columns.iter().into());
        TrainResultManager {
            iteration: 0,
            next_output_iteration: 0,
            period,
            last_render_height: 0,
            table,
            columns,
        }
    }

    #[must_use = "Output indicates if results should be added for this iteration."]
    fn inc_iteration(&mut self) -> bool {
        self.iteration += 1;
        if self.iteration >= self.next_output_iteration {
            self.next_output_iteration = match self.period {
                OutputPeriod::Additive(period) => self.iteration + period,
                OutputPeriod::Multiplicative(period) => (self.iteration as f32 * period) as i32,
            };
            return true;
        }
        false
    }

    fn add_results(&mut self, results: Vec<MetricValue>) {
        if results.len() == self.columns.len() {
            self.table
                .add_row(results.iter().map(|v| v.to_string()).into());
        } else {
            panic!("Results columns do not match previous results columns.");
        }
    }

    fn render_table_to_stdout(&mut self) {
        let mut stdout = term::stdout().unwrap();
        for _ in 0..self.last_render_height {
            stdout.cursor_up().unwrap();
            stdout.delete_line().unwrap();
        }
        self.last_render_height = self.table.print_tty(false).unwrap() as u16;
    }
}

pub(crate) struct TrainCommand;

impl Command for TrainCommand {
    type Args = TrainArgs;
    fn execute(args: &TrainArgs, quiet: bool) -> Result<()> {
        if let Some(size) = args.thread_pool_size {
            rayon::ThreadPoolBuilder::new()
                .num_threads(size)
                .build_global()?;
        }

        let mut workspace = match (&args.input_config.config, &args.input_config.input_model) {
            // Loading from json config
            (Some(config_file), None) => {
                let json = std::fs::read_to_string(config_file).unwrap();
                reductionml_core::workspace::Workspace::create_from_json(&json).unwrap()
            }
            // Loading from model file
            (None, Some(input_model_file)) => {
                let data = std::fs::read(input_model_file).unwrap();
                reductionml_core::workspace::Workspace::create_from_model(&data).unwrap()
            }
            _ => unreachable!(),
        };

        let file = File::open(&args.data).unwrap();

        eprintln!(
            "{}: Reading data file: {}",
            "info".cyan().bold(),
            &args.data.bold()
        );
        eprintln!(
            "{}: Using entry reduction: {}",
            "info".cyan().bold(),
            workspace.get_entry_reduction().typename().bold()
        );
        eprintln!("{}: Starting training...", "info".cyan().bold());

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

        let mut input_file = io::BufReader::new(file);
        let mut predictions_file = if args.predictions.is_some() {
            eprintln!(
                "{}: The format output in the predictions file is currently a placeholder",
                "warning".yellow().bold()
            );
            let file = File::create(&args.predictions.as_ref().unwrap()).unwrap();
            Some(io::BufWriter::new(file))
        } else {
            None
        };

        let mut metrics: Vec<Box<dyn Metric>> = if (args.metrics.is_some()) {
            args.metrics
                .as_ref()
                .unwrap()
                .iter()
                .map(|name| reductionml_core::metrics::get_metric(&name).unwrap())
                .collect()
        } else {
            vec![]
        };

        let mut col_names: Vec<String> = vec!["Example #".into()];
        col_names.extend(metrics.iter().map(|x| x.get_name()));
        let mut manager = TrainResultManager::new(OutputPeriod::Multiplicative(2.0), col_names);

        let (tx, rx) = flume::bounded(args.queue_size);

        let string_pool = object_pool::Pool::<String>::new();

        std::thread::scope(|s| -> Result<()> {
            s.spawn(|| {
                loop {
                    let mut batch = vec![];
                    for _ in 0..args.read_batch_size {
                        if let Some(chunk) = parser
                            .get_next_chunk(&mut input_file, string_pool.get_object())
                            .unwrap()
                        {
                            batch.push(chunk);
                        } else {
                            break;
                        }
                    }

                    if batch.is_empty() {
                        break;
                    }

                    // Must be collected so that original order is preserved.
                    let batch: Vec<(Features<'_>, Option<Label>)> = batch
                        .into_par_iter()
                        .map(|line| {
                            let res = parser.parse_chunk(&line).unwrap();
                            string_pool.return_object(line);
                            res
                        })
                        .collect();

                    for item in batch {
                        tx.send(item).expect(
                            "Receiver should not be disconnected before all lines have been sent.",
                        );
                    }
                }
                std::mem::drop(tx);
            });

            loop {
                let res = rx.recv();
                match res {
                    Ok((features, label)) => {
                        let prediction = workspace.predict(&features);
                        if let Some(file) = predictions_file.as_mut() {
                            // TODO: some cannonical format for prediction values.
                            writeln!(file, "{:?}", prediction).unwrap();
                        }

                        let label = label.unwrap();
                        workspace.learn(&features, &label);

                        for metric in metrics.iter_mut() {
                            metric.add_point(&features, &label, &prediction);
                        }

                        counter += 1;
                        let should_output = manager.inc_iteration();
                        if should_output {
                            let mut results = vec![MetricValue::Int(counter)];
                            results.extend(metrics.iter().map(|x| x.get_value()));
                            manager.add_results(results.into());
                            manager.render_table_to_stdout();
                        }

                        // Put feature objects back into the pool for reuse.
                        features.clear_and_return_object(pool.as_ref());
                    }
                    Err(_) => break,
                }
            }
            Ok(())
        })?;

        let mut results = vec![MetricValue::Int(counter)];
        results.extend(metrics.iter().map(|x| x.get_value()));
        manager.add_results(results.into());
        manager.render_table_to_stdout();

        Ok(())
    }
}
