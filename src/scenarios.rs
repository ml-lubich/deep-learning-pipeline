//! Shared flows for the CLI, example binary, and integration tests.

use crate::datasets::Dataset;
use crate::model::LinearModel;
use crate::train::{fit_sgd, mean_squared_error};

/// Deterministic narrative report for demos and CLI output assertions.
#[must_use]
pub fn run_demo_report() -> String {
    let dataset = Dataset::synthetic_linear(96, 42);
    let (train, val) = dataset.split(0.75).expect("split");
    let mut model = LinearModel::new(2);
    let _losses = fit_sgd(&mut model, train.rows(), 320, 0.06).expect("fit");
    let train_mse = mean_squared_error(&model, train.rows()).expect("train mse");
    let val_mse = mean_squared_error(&model, val.rows()).expect("val mse");
    format!(
        "deep learning pipeline sample\n  train_rows={}\n  val_rows={}\n  train_mse={train_mse:.5}\n  val_mse={val_mse:.5}\n",
        train.len(),
        val.len(),
    )
}
