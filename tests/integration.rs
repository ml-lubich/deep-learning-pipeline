//! Integration tests: dataset split, training loop, validation metric.

use deep_learning_pipeline::{Dataset, DatasetError, LinearModel, fit_sgd, mean_squared_error};

#[test]
fn split_train_and_val_both_non_empty() {
    let ds = Dataset::synthetic_linear(32, 1);
    let (train, val) = ds.split(0.75).expect("split");
    assert!(!train.is_empty());
    assert!(!val.is_empty());
}

#[test]
fn train_loop_improves_mse_on_synthetic() {
    let ds = Dataset::synthetic_linear(40, 21);
    let (train, _) = ds.split(0.8).expect("split");
    let mut model = LinearModel::new(2);
    let before = mean_squared_error(&model, train.rows()).expect("mse");
    fit_sgd(&mut model, train.rows(), 300, 0.05).expect("fit");
    let after = mean_squared_error(&model, train.rows()).expect("mse");
    assert!(after < before);
}

#[test]
fn invalid_split_ratio_is_rejected() {
    let ds = Dataset::synthetic_linear(5, 1);
    assert_eq!(ds.split(0.0).unwrap_err(), DatasetError::InvalidSplitRatio);
}
