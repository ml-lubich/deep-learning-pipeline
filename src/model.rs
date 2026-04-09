//! Minimal linear model with manual SGD updates (no external ML framework).

use crate::error::TrainError;

/// Single-layer linear regressor: `y = w·x + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearModel {
    weights: Vec<f32>,
    bias: f32,
}

impl LinearModel {
    /// Small non-zero initialization (deterministic constant for reproducibility in tests).
    #[must_use]
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: vec![0.01; input_dim],
            bias: 0.0,
        }
    }

    /// Input dimension (length of `weights`).
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.weights.len()
    }

    /// Forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::DimMismatch`] when `features.len()` ≠ input dimension.
    pub fn forward(&self, features: &[f32]) -> Result<f32, TrainError> {
        let expected = self.weights.len();
        let got = features.len();
        if expected != got {
            return Err(TrainError::DimMismatch { expected, got });
        }
        let mut sum = self.bias;
        for (w, x) in self.weights.iter().zip(features.iter()) {
            sum = w.mul_add(*x, sum);
        }
        Ok(sum)
    }

    /// One full-gradient SGD step on a single example; returns scalar MSE contribution `0.5 * err²`.
    ///
    /// # Errors
    ///
    /// Propagates [`TrainError::DimMismatch`] from [`Self::forward`].
    pub fn train_step(&mut self, features: &[f32], label: f32, lr: f32) -> Result<f32, TrainError> {
        let pred = self.forward(features)?;
        let err = pred - label;
        for (w, x) in self.weights.iter_mut().zip(features.iter()) {
            *w -= lr * err * x;
        }
        self.bias -= lr * err;
        Ok(0.5 * err * err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_rejects_wrong_dim() {
        let m = LinearModel::new(2);
        let err = m.forward(&[1.0]).expect_err("dim");
        assert_eq!(
            err,
            TrainError::DimMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn train_step_runs_on_correct_dim() {
        let mut m = LinearModel::new(2);
        let loss = m.train_step(&[0.0, 0.0], 0.0, 0.1).expect("step");
        assert!(loss.is_finite());
    }
}
