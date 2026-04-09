//! Stochastic training loops over in-memory rows.

use crate::datasets::Row;
use crate::error::TrainError;
use crate::model::LinearModel;

/// Stochastic gradient descent over `rows`, cycling indices for `steps` updates.
///
/// # Errors
///
/// Propagates the first [`TrainError`] from [`LinearModel::train_step`].
pub fn fit_sgd(
    model: &mut LinearModel,
    rows: &[Row],
    steps: usize,
    lr: f32,
) -> Result<Vec<f32>, TrainError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let mut losses = Vec::with_capacity(steps);
    for t in 0..steps {
        let row = &rows[t % rows.len()];
        let loss = model.train_step(&row.features, row.label, lr)?;
        losses.push(loss);
    }
    Ok(losses)
}

/// Mean squared error on a slice of labeled rows.
///
/// # Errors
///
/// Propagates [`TrainError::DimMismatch`] from [`LinearModel::forward`].
pub fn mean_squared_error(model: &LinearModel, rows: &[Row]) -> Result<f32, TrainError> {
    if rows.is_empty() {
        return Ok(0.0);
    }
    let mut acc = 0.0_f32;
    for row in rows {
        let pred = model.forward(&row.features)?;
        let err = pred - row.label;
        acc += err * err;
    }
    let n = rows.len() as f32;
    Ok(acc / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::Dataset;

    #[test]
    fn fit_reduces_loss_on_linear_synthetic() {
        let ds = Dataset::synthetic_linear(48, 99);
        let (train, _) = ds.split(0.85).expect("split");
        let mut model = LinearModel::new(2);
        let before = mean_squared_error(&model, train.rows()).expect("mse");
        let losses = fit_sgd(&mut model, train.rows(), 400, 0.08).expect("fit");
        let after = mean_squared_error(&model, train.rows()).expect("mse");
        assert!(after < before);
        assert!(losses.last().is_some_and(|l| l.is_finite()));
    }
}
