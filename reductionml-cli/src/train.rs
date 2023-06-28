use std::{
    fs::File,
    io::{self, stdout, Write},
    str::FromStr,
};

use anyhow::{Context, Result};

use clap::{Args, ValueHint};
use owo_colors::OwoColorize;
// use crossterm::{cursor, terminal, ExecutableCommand};

use crossterm::{cursor, terminal, ExecutableCommand};
use prettytable::{format, Table};
use reductionml_core::{
    metrics::{Metric, MetricValue},
    object_pool::{self, PoolReturnable},
    Features, Label,
};

use crate::{command::Command, DataFormat, InputConfigArg};
use rayon::prelude::*;

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

    #[arg(long)]
    #[arg(default_value = "*2")]
    #[arg(value_parser = clap::value_parser!(OutputPeriod))]
    progress: OutputPeriod,

    // Save final model to file
    #[arg(short, long)]
    output_model: Option<String>,

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
    /// When this is 0, a single thread will be used for parsing+training.
    /// When this is 1, one thread will be used for parsing and one for training.
    #[arg(long)]
    #[arg(default_value = None)]
    thread_pool_size: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum OutputPeriod {
    Additive(u32),
    Multiplicative(f32),
}

impl FromStr for OutputPeriod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.to_lowercase();
        if let Some(stripped) = value.strip_prefix('+') {
            let period = stripped.parse::<u32>().map_err(|_| {
                format!(
                    "Invalid output period: {}. Must be of the form <int>, +<int> or *<float>",
                    value
                )
            })?;
            Ok(OutputPeriod::Additive(period))
        } else if let Some(stripped) = value.strip_prefix('*') {
            let period = stripped.parse::<f32>().map_err(|_| {
                format!(
                    "Invalid output period: {}. Must be of the form <int>, +<int> or *<float>",
                    value
                )
            })?;
            Ok(OutputPeriod::Multiplicative(period))
        } else {
            if let Ok(period) = value.parse::<u32>() {
                return Ok(OutputPeriod::Additive(period));
            }

            Err(format!(
                "Invalid output period: {}. Must be of the form <int>, +<int> or *<float>",
                value
            ))
        }
    }
}

struct TrainResultManager {
    iteration: u32,
    next_output_iteration: u32,
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
        if self.iteration >= self.next_output_iteration {
            self.next_output_iteration = match self.period {
                OutputPeriod::Additive(period) => self.iteration + period,
                OutputPeriod::Multiplicative(period) => (self.iteration as f32 * period) as u32,
            };
            self.iteration += 1;
            return true;
        }
        self.iteration += 1;
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

    // FIXME: if the period is too low and the training too fast then this will cause rendering issues.
    fn render_table_to_stdout(&mut self) {
        let mut stdout = stdout();
        stdout
            .execute(cursor::MoveUp(self.last_render_height))
            .unwrap();
        stdout
            .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
            .unwrap();
        self.last_render_height = self.table.print_tty(false).unwrap() as u16;
    }
}

pub(crate) struct TrainCommand;

impl Command for TrainCommand {
    type Args = TrainArgs;
    fn execute(args: &TrainArgs, _quiet: bool) -> Result<()> {
        if let Some(size) = args.thread_pool_size {
            rayon::ThreadPoolBuilder::new()
                .num_threads(size)
                .build_global()?;
        }

        let mut workspace = match (&args.input_config.config, &args.input_config.input_model) {
            // Loading from json config
            (Some(config_file), None) => {
                let json = std::fs::read_to_string(config_file)
                    .with_context(|| format!("Failed to read config file: {}", config_file))?;
                reductionml_core::workspace::Workspace::create_from_json(&json).with_context(
                    || {
                        format!(
                            "Failed to create workspace from config file: {}",
                            config_file
                        )
                    },
                )?
            }
            // Loading from model file
            (None, Some(input_model_file)) => {
                let data = std::fs::read(input_model_file).with_context(|| {
                    format!("Failed to read input model file: {}", input_model_file)
                })?;
                reductionml_core::workspace::Workspace::create_from_model(&data).with_context(
                    || {
                        format!(
                            "Failed to create workspace from input model file: {}",
                            input_model_file
                        )
                    },
                )?
            }
            _ => unreachable!(),
        };

        let file = File::open(&args.data)
            .with_context(|| format!("Failed to open data file: {}", args.data))?;
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

        let pool = workspace.features_pool().clone();

        let parser = args.data_format.get_parser(
            workspace
                .get_entry_reduction()
                .types()
                .input_features_type(),
            workspace.get_entry_reduction().types().input_label_type(),
            workspace.global_config().hash_seed(),
            workspace.global_config().num_bits(),
            pool.clone(),
        );

        let mut input_file = io::BufReader::new(file);
        let mut predictions_file = if let Some(pred_file_name) = &args.predictions {
            eprintln!(
                "{}: The format output in the predictions file is currently a placeholder",
                "warning".yellow().bold()
            );
            let file = File::create(pred_file_name).with_context(|| {
                format!("Failed to create predictions file: {}", pred_file_name)
            })?;
            Some(io::BufWriter::new(file))
        } else {
            None
        };

        let mut metrics = vec![reductionml_core::metrics::get_metric("example_number").unwrap()];
        if args.metrics.is_some() {
            args.metrics.as_ref().unwrap().iter().for_each(|name| {
                metrics.push(reductionml_core::metrics::get_metric(name).unwrap())
            });
        };

        let mut manager = TrainResultManager::new(
            args.progress,
            metrics.iter().map(|x| x.get_name()).collect(),
        );

        match args.thread_pool_size {
            Some(0) => {
                let mut buffer = String::new();
                while let Some(chunk) = parser.get_next_chunk(&mut input_file, buffer).unwrap() {
                    let (features, label) = parser.parse_chunk(&chunk).unwrap();
                    buffer = chunk;
                    process_example(
                        label,
                        &mut workspace,
                        features,
                        &mut predictions_file,
                        &mut metrics,
                        &mut manager,
                        _quiet,
                        &pool,
                    );
                }
            }
            Some(1) => {
                let (tx, rx) = flume::bounded(args.queue_size);
                std::thread::scope(|s| -> Result<()> {
                    s.spawn(|| {
                        let mut buffer = String::new();
                        while let Some(chunk) =  parser
                        .get_next_chunk(&mut input_file, buffer)
                        .unwrap() {
                            let result = parser.parse_chunk(&chunk).unwrap();
                            tx.send(result).expect(
                                "Receiver should not be disconnected before all lines have been sent.",
                            );
                            buffer = chunk;
                        }
                        std::mem::drop(tx);
                    });

                    loop {
                        let res = rx.recv();
                        match res {
                            Ok((features, label)) => {
                                process_example(
                                    label,
                                    &mut workspace,
                                    features,
                                    &mut predictions_file,
                                    &mut metrics,
                                    &mut manager,
                                    _quiet,
                                    &pool,
                                );
                            }
                            Err(_) => break,
                        }
                    }
                    Ok(())
                })?;
            }
            _ => {
                let string_pool = object_pool::Pool::<String>::new();
                let (tx, rx) = flume::bounded(args.queue_size);
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
                                process_example(
                                    label,
                                    &mut workspace,
                                    features,
                                    &mut predictions_file,
                                    &mut metrics,
                                    &mut manager,
                                    _quiet,
                                    &pool,
                                );
                            }
                            Err(_) => break,
                        }
                    }
                    Ok(())
                })?;
            }
        }

        manager.add_results(metrics.iter().map(|x| x.get_value()).collect());
        if !_quiet {
            manager.render_table_to_stdout();
        }

        if let Some(file) = &args.output_model {
            let data = workspace.serialize_model().unwrap();
            std::fs::write(file, data).unwrap();
        }

        Ok(())
    }
}

fn process_example(
    label: Option<Label>,
    workspace: &mut reductionml_core::workspace::Workspace,
    features: Features<'_>,
    predictions_file: &mut Option<io::BufWriter<File>>,
    metrics: &mut Vec<Box<dyn Metric>>,
    manager: &mut TrainResultManager,
    _quiet: bool,
    pool: &std::sync::Arc<
        object_pool::Pool<reductionml_core::sparse_namespaced_features::SparseFeatures>,
    >,
) {
    let label = label.unwrap();
    let prediction = workspace.predict_then_learn(&features, &label);
    if let Some(file) = predictions_file.as_mut() {
        writeln!(file, "{}", serde_json::to_string(&prediction).unwrap()).unwrap();
    }

    for metric in metrics.iter_mut() {
        metric.add_point(&features, &label, &prediction);
    }

    let should_output = manager.inc_iteration();
    if should_output {
        manager.add_results(metrics.iter().map(|x| x.get_value()).collect());
        if !_quiet {
            manager.render_table_to_stdout();
        }
    }

    // Put feature objects back into the pool for reuse.
    features.clear_and_return_object(pool.as_ref());
}
