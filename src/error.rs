//! Domain errors for datasets and training.

use thiserror::Error;

/// Failures while building or splitting datasets.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum DatasetError {
    /// Ratio must lie strictly between 0 and 1.
    #[error("train/validation split ratio must be in (0,1)")]
    InvalidSplitRatio,
    /// No rows to split or train on.
    #[error("dataset is empty")]
    Empty,
    /// At least two rows are required to form both train and validation sets.
    #[error("need at least two rows to split into train and validation")]
    TooSmallForSplit,
}

/// Failures during forward passes or optimization steps.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum TrainError {
    /// Feature vector length does not match model input size.
    #[error("feature dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected feature length (model input size).
        expected: usize,
        /// Actual feature length supplied to [`crate::model::LinearModel::forward`].
        got: usize,
    },
}
