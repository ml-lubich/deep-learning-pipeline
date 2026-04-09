//! `dlpipe` CLI — in-process harness for datasets and the tiny linear trainer.

use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use deep_learning_pipeline::art::{HELP_LONG, TAGLINE};
use deep_learning_pipeline::datasets::Dataset;
use deep_learning_pipeline::model::LinearModel;
use deep_learning_pipeline::scenarios;
use deep_learning_pipeline::train::{fit_sgd, mean_squared_error};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "dlpipe",
    version,
    about = TAGLINE,
    long_about = HELP_LONG
)]
struct Cli {
    /// Enable `tracing` spans/events (honors `RUST_LOG`, default `info` when unset).
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the scripted dataset → train → validation report.
    Demo,
    /// Quick fit on a synthetic set (`rows`, `steps`, learning rate).
    Fit {
        /// Number of synthetic examples.
        #[arg(short, long, default_value_t = 64)]
        rows: usize,
        /// SGD steps.
        #[arg(short, long, default_value_t = 200)]
        steps: usize,
        /// Learning rate.
        #[arg(short, long, default_value_t = 0.07)]
        lr: f32,
    },
}

fn init_tracing(verbose: bool) -> Result<()> {
    if !verbose {
        return Ok(());
    }

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init()
        .map_err(|e| anyhow::anyhow!("failed to install tracing subscriber: {e}"))?;

    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Demo => {
            let report = scenarios::run_demo_report();
            print!("{report}");
        }
        Command::Fit { rows, steps, lr } => {
            let n = rows.max(4);
            let ds = Dataset::synthetic_linear(n, 11);
            let (train, val) = ds.split(0.75).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut model = LinearModel::new(2);
            fit_sgd(&mut model, train.rows(), steps, lr).map_err(|e| anyhow::anyhow!("{e}"))?;
            let train_mse =
                mean_squared_error(&model, train.rows()).map_err(|e| anyhow::anyhow!("{e}"))?;
            let val_mse =
                mean_squared_error(&model, val.rows()).map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("rows={n} train_mse={train_mse:.5} val_mse={val_mse:.5}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(err) = init_tracing(cli.verbose) {
        eprintln!("{err:#}");
        return ExitCode::FAILURE;
    }
    if let Err(err) = run(cli) {
        tracing::error!(error = %err, "dlpipe failed");
        eprintln!("{err:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
