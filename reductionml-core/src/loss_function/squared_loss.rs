use approx::{assert_abs_diff_eq, assert_relative_eq};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::loss_function::LossFunctionImpl;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, Default)]
pub struct SquaredLoss {}

impl SquaredLoss {
    pub(crate) fn new() -> SquaredLoss {
        SquaredLoss {}
    }
}

impl LossFunctionImpl for SquaredLoss {
    fn get_loss(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32 {
        if prediction <= max_label && prediction >= min_label {
            (prediction - label) * (prediction - label)
        } else if prediction < min_label {
            if label == min_label {
                return 0.;
            } else {
                return (label - min_label) * (label - min_label)
                    + 2. * (label - min_label) * (min_label - prediction);
            }
        } else if label == max_label {
            return 0.;
        } else {
            return (max_label - label) * (max_label - label)
                + 2. * (max_label - label) * (prediction - max_label);
        }
    }

    fn get_update(
        &self,
        prediction: f32,
        label: f32,
        update_scale: f32,
        pred_per_update: f32,
    ) -> f32 {
        if update_scale * pred_per_update < 1e-6 {
            /* When exp(-eta_t)~= 1 we replace 1-exp(-eta_t)
             * with its first order Taylor expansion around 0
             * to avoid catastrophic cancellation.
             */
            return 2.0 * (label - prediction) * update_scale;
        }

        (label - prediction) * (1.0 - (-2.0 * update_scale * pred_per_update).exp())
            / pred_per_update
    }

    fn get_unsafe_update(&self, prediction: f32, label: f32, update_scale: f32) -> f32 {
        2.0 * (label - prediction) * update_scale
    }

    fn get_square_grad(&self, prediction: f32, label: f32) -> f32 {
        4.0 * (prediction - label) * (prediction - label)
    }

    fn first_derivative(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32 {
        let pred = if prediction < min_label {
            min_label
        } else if prediction > max_label {
            max_label
        } else {
            prediction
        };

        2.0 * (pred - label)
    }

    fn second_derivative(
        &self,
        min_label: f32,
        max_label: f32,
        prediction: f32,
        _label: f32,
    ) -> f32 {
        if prediction <= max_label && prediction >= min_label {
            2.
        } else {
            0.
        }
    }
}

#[test]
fn squared_loss_test() {
    let loss_function = SquaredLoss::new();

    let min_label = 0.0;
    let max_label = 1.0;

    let learning_rate = 0.1;
    let weight = 1.0;

    let label = 0.5;
    let prediction = 0.4;
    let update_scale = learning_rate * weight;
    let pred_per_update = 1.0;

    assert_relative_eq!(
        0.01,
        loss_function.get_loss(min_label, max_label, prediction, label)
    );
    assert_relative_eq!(
        0.01812692,
        loss_function.get_update(prediction, label, update_scale, pred_per_update)
    );
    assert_relative_eq!(
        0.02,
        loss_function.get_unsafe_update(prediction, label, update_scale)
    );

    assert_relative_eq!(0.04, loss_function.get_square_grad(prediction, label));
    assert_relative_eq!(
        -0.2,
        loss_function.first_derivative(min_label, max_label, prediction, label)
    );
    assert_relative_eq!(
        2.0,
        loss_function.second_derivative(min_label, max_label, prediction, label)
    );
}
