use atomic_wait::{wait, wake_one};
use reductionml_core::workspace::Configuration;
use std::sync::atomic::AtomicU32;
use std::{
    cell::UnsafeCell,
    fs::File,
    io::{self, stdout, Write},
    str::FromStr,
    sync::{atomic::Ordering, Arc},
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
    #[arg(default_value = "512")]
    queue_size: usize,

    /// Number of threads to use for the rayon thread pool. By default will use
    /// the number of logical cores - 2.
    /// When this is 0, a single thread will be used for parsing+training.
    #[arg(long)]
    #[arg(default_value = None)]
    num_parse_threads: Option<usize>,
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

struct ParseResult<'a> {
    // 0 not ready, 1 ready
    ready: AtomicU32,
    result: UnsafeCell<Option<Result<(Features<'a>, Option<Label>)>>>,
    input: UnsafeCell<Option<String>>,
}

impl<'a> ParseResult<'a> {
    fn new(input: String) -> ParseResult<'a> {
        ParseResult {
            ready: AtomicU32::new(0),
            result: UnsafeCell::new(None),
            input: UnsafeCell::new(Some(input)),
        }
    }

    fn get_input(&self) -> String {
        assert!(self.ready.load(Ordering::SeqCst) == 0);
        unsafe { (*self.input.get()).take().unwrap() }
    }

    fn await_result(&self) -> Result<(Features<'a>, Option<Label>)> {
        while self.ready.load(Ordering::Relaxed) == 0 {
            wait(&self.ready, 0);
        }
        assert!(self.ready.load(Ordering::SeqCst) == 1);
        unsafe { (*self.result.get()).take().unwrap() }
    }

    fn set_result(&self, result: Result<(Features<'a>, Option<Label>)>) {
        assert!(self.ready.load(Ordering::SeqCst) == 0);
        unsafe {
            *self.result.get() = Some(result);
        }
        self.ready.store(1, Ordering::SeqCst);
        wake_one(&self.ready);
    }
}

unsafe impl Sync for ParseResult<'_> {}

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
    fn execute(args: &TrainArgs, quiet: bool) -> Result<()> {
        let mut workspace = match (&args.input_config.config, &args.input_config.input_model) {
            // Loading from json config
            (Some(config_file), None) => {
                let json = std::fs::read_to_string(config_file)
                    .with_context(|| format!("Failed to read config file: {}", config_file))?;
                reductionml_core::workspace::Workspace::new(Configuration::from_json_str(&json)?)
                    .with_context(|| {
                        format!(
                            "Failed to create workspace from config file: {}",
                            config_file
                        )
                    })?
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

        let num_parse_threads = match args.num_parse_threads {
            Some(n) => n,
            None => (num_cpus::get() as i32 - 2).max(0) as usize,
        };

        match num_parse_threads {
            0 => {
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
                        quiet,
                        &pool,
                    );
                }
            }
            n => {
                let string_pool = object_pool::Pool::<String>::new();
                let (parse_sender, parse_receiver) = flume::bounded(args.queue_size);
                let (learn_sender, learn_receiver) = flume::bounded(args.queue_size);
                std::thread::scope(|s| -> Result<()> {
                    // Input thread
                    s.spawn(|| {
                        loop {
                            if let Some(chunk) = parser
                                .get_next_chunk(&mut input_file, string_pool.get_object())
                                .unwrap()
                            {
                                let res = Arc::new(ParseResult::new(chunk));
                                parse_sender.send(res.clone()).unwrap();
                                learn_sender.send(res).unwrap();
                            } else {
                                break;
                            }
                        }
                        std::mem::drop(parse_sender);
                        std::mem::drop(learn_sender);
                    });

                    for _ in 0..n {
                        s.spawn(|| loop {
                            match parse_receiver.recv() {
                                Ok(res) => {
                                    let input = res.get_input();
                                    let parsed = parser.parse_chunk(&input);
                                    string_pool.return_object(input);
                                    res.set_result(parsed.map_err(|e| anyhow::anyhow!(e)));
                                }
                                Err(_) => break,
                            }
                        });
                    }

                    loop {
                        let res = learn_receiver.recv();
                        match res {
                            Ok(result) => {
                                let (features, label) = result.await_result().unwrap();
                                process_example(
                                    label,
                                    &mut workspace,
                                    features,
                                    &mut predictions_file,
                                    &mut metrics,
                                    &mut manager,
                                    quiet,
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

        if !quiet {
            manager.add_results(metrics.iter().map(|x| x.get_value()).collect());
            manager.render_table_to_stdout();
        }

        if let Some(file) = &args.output_model {
            let data = workspace.serialize_model().unwrap();
            std::fs::write(file, data).unwrap();
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn process_example(
    label: Option<Label>,
    workspace: &mut reductionml_core::workspace::Workspace,
    mut features: Features<'_>,
    predictions_file: &mut Option<io::BufWriter<File>>,
    metrics: &mut [Box<dyn Metric>],
    manager: &mut TrainResultManager,
    quiet: bool,
    pool: &std::sync::Arc<
        object_pool::Pool<reductionml_core::sparse_namespaced_features::SparseFeatures>,
    >,
) {
    let label = label.unwrap();
    if !quiet || predictions_file.is_some() {
        let prediction = workspace.predict_then_learn(&mut features, &label);
        if let Some(file) = predictions_file.as_mut() {
            writeln!(file, "{}", serde_json::to_string(&prediction).unwrap()).unwrap();
        }

        for metric in metrics.iter_mut() {
            metric.add_point(&features, &label, &prediction);
        }

        let should_output = manager.inc_iteration();
        if should_output {
            manager.add_results(metrics.iter().map(|x| x.get_value()).collect());
            manager.render_table_to_stdout();
        }
    } else {
        workspace.learn(&mut features, &label);
    }

    // Put feature objects back into the pool for reuse.
    features.clear_and_return_object(pool.as_ref());
}
