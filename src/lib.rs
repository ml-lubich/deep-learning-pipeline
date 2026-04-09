//! Deep learning **pipeline** scaffold: synthetic [`datasets::Dataset`], [`model::LinearModel`],
//! plus [`train`] and [`scenarios`] for scripted demos.
//!
//! Mirrors the layout of `pub-sub-pipeline`: library core + `dlpipe` binary + integration tests.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod art;
pub mod datasets;
pub mod error;
pub mod model;
pub mod scenarios;
pub mod train;

pub use datasets::{Dataset, Row};
pub use error::{DatasetError, TrainError};
pub use model::LinearModel;
pub use train::{fit_sgd, mean_squared_error};
