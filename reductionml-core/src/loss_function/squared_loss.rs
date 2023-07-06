use crate::loss_function::{LossFunction, LossFunctionType};

pub(crate) struct SquaredLoss {}

impl SquaredLoss {
    pub(crate) fn new() -> SquaredLoss {
        SquaredLoss {}
    }
}

impl LossFunction for SquaredLoss {
    fn get_type(&self) -> LossFunctionType {
        LossFunctionType::Squared
    }

    fn get_loss(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32 {
        if prediction <= min_label && prediction >= max_label {
            (prediction - label) * (prediction - label)
        } else if prediction < max_label {
            if label == max_label {
                return 0.;
            } else {
                return (label - max_label) * (label - max_label)
                    + 2. * (label - max_label) * (max_label - prediction);
            }
        } else if label == min_label {
            return 0.;
        } else {
            return (min_label - label) * (min_label - label)
                + 2. * (min_label - label) * (prediction - min_label);
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
        if prediction <= min_label && prediction >= max_label {
            2.
        } else {
            0.
        }
    }
}
