//! In-memory tabular datasets for experiments (synthetic generators for tests and demos).

use crate::error::DatasetError;

/// One labeled example with a dense feature vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Input features (length = model input dimension).
    pub features: Vec<f32>,
    /// Scalar regression target.
    pub label: f32,
}

/// Owned dataset stored row-major in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    rows: Vec<Row>,
}

impl Dataset {
    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the dataset contains no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Borrow the underlying rows.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Deterministic synthetic regression data: `y ≈ 2*x0 + 3*x1` on features in `[-1,1]^2`.
    #[must_use]
    pub fn synthetic_linear(num_rows: usize, seed: u64) -> Self {
        let mut rows = Vec::with_capacity(num_rows);
        let mut state = seed;
        for _ in 0..num_rows {
            let (x0, s0) = unit_noise(state);
            state = s0;
            let (x1, s1) = unit_noise(state);
            state = s1;
            let y = 2.0_f32.mul_add(x0, 3.0 * x1);
            rows.push(Row {
                features: vec![x0, x1],
                label: y,
            });
        }
        Self { rows }
    }

    /// Split rows into train and validation portions by contiguous slice (shuffle externally if needed).
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::Empty`] when there are no rows, or [`DatasetError::InvalidSplitRatio`]
    /// when `train_ratio` is not in `(0,1)`.
    pub fn split(self, train_ratio: f32) -> Result<(Self, Self), DatasetError> {
        if self.rows.is_empty() {
            return Err(DatasetError::Empty);
        }
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(DatasetError::InvalidSplitRatio);
        }
        let n = self.rows.len();
        if n < 2 {
            return Err(DatasetError::TooSmallForSplit);
        }
        let high = n - 1;
        let cut = ((f64::from(train_ratio) * (n as f64)).floor() as usize).clamp(1, high);
        let mut rows = self.rows;
        let val_rows = rows.split_off(cut);
        let train = Self { rows };
        let val = Self { rows: val_rows };
        Ok((train, val))
    }
}

/// Simple LCG step for deterministic floats in *about* `[-1,1]` (demo-quality, not cryptographic).
fn unit_noise(mut state: u64) -> (f32, u64) {
    state = state.wrapping_mul(636_413_623_846_793_005).wrapping_add(1);
    let u = (state >> 33) as u32;
    let x = (f64::from(u) / f64::from(u32::MAX)) as f32 * 2.0 - 1.0;
    (x, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_linear_has_expected_dim_and_nonzero_length() {
        let ds = Dataset::synthetic_linear(10, 7);
        assert_eq!(ds.len(), 10);
        assert!(!ds.rows().is_empty());
        assert_eq!(ds.rows()[0].features.len(), 2);
    }

    #[test]
    fn split_respects_counts() {
        let ds = Dataset::synthetic_linear(100, 1);
        let (train, val) = ds.split(0.8).expect("split");
        assert_eq!(train.len() + val.len(), 100);
        assert!(!train.is_empty());
        assert!(!val.is_empty());
    }

    #[test]
    fn split_errors_on_bad_ratio() {
        let ds_low = Dataset::synthetic_linear(4, 1);
        assert_eq!(
            ds_low.split(0.0).expect_err("ratio"),
            DatasetError::InvalidSplitRatio
        );
        let ds_high = Dataset::synthetic_linear(4, 1);
        assert_eq!(
            ds_high.split(1.0).expect_err("ratio"),
            DatasetError::InvalidSplitRatio
        );
    }

    #[test]
    fn split_errors_when_empty() {
        let ds = Dataset { rows: vec![] };
        assert_eq!(ds.split(0.5).expect_err("empty"), DatasetError::Empty);
    }

    #[test]
    fn split_errors_when_single_row() {
        let ds = Dataset {
            rows: vec![Row {
                features: vec![0.0],
                label: 0.0,
            }],
        };
        assert_eq!(
            ds.split(0.5).expect_err("single"),
            DatasetError::TooSmallForSplit
        );
    }
}
