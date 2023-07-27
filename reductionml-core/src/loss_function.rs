mod logistic_loss;
mod squared_loss;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use self::logistic_loss::*;
pub use self::squared_loss::*;

pub trait LossFunctionImpl: Send {
    fn get_loss(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32;
    fn get_update(
        &self,
        prediction: f32,
        label: f32,
        update_scale: f32,
        pred_per_update: f32,
    ) -> f32;
    fn get_unsafe_update(&self, prediction: f32, label: f32, update_scale: f32) -> f32;
    fn get_square_grad(&self, prediction: f32, label: f32) -> f32;
    fn first_derivative(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32;
    fn second_derivative(&self, min_label: f32, max_label: f32, prediction: f32, label: f32)
        -> f32;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
pub enum LossFunction {
    Squared(SquaredLoss),
    Logistic(LogisticLoss),
}

impl From<SquaredLoss> for LossFunction {
    fn from(loss: SquaredLoss) -> Self {
        LossFunction::Squared(loss)
    }
}

impl TryFrom<LossFunction> for SquaredLoss {
    type Error = ();

    fn try_from(loss: LossFunction) -> Result<Self, Self::Error> {
        match loss {
            LossFunction::Squared(loss) => Ok(loss),
            _ => Err(()),
        }
    }
}

impl From<LogisticLoss> for LossFunction {
    fn from(loss: LogisticLoss) -> Self {
        LossFunction::Logistic(loss)
    }
}

impl TryFrom<LossFunction> for LogisticLoss {
    type Error = ();

    fn try_from(loss: LossFunction) -> Result<Self, Self::Error> {
        match loss {
            LossFunction::Logistic(loss) => Ok(loss),
            _ => Err(()),
        }
    }
}

impl LossFunctionImpl for LossFunction {
    fn get_loss(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32 {
        match self {
            LossFunction::Squared(loss) => loss.get_loss(min_label, max_label, prediction, label),
            LossFunction::Logistic(loss) => loss.get_loss(min_label, max_label, prediction, label),
        }
    }

    fn get_update(
        &self,
        prediction: f32,
        label: f32,
        update_scale: f32,
        pred_per_update: f32,
    ) -> f32 {
        match self {
            LossFunction::Squared(loss) => {
                loss.get_update(prediction, label, update_scale, pred_per_update)
            }
            LossFunction::Logistic(loss) => {
                loss.get_update(prediction, label, update_scale, pred_per_update)
            }
        }
    }

    fn get_unsafe_update(&self, prediction: f32, label: f32, update_scale: f32) -> f32 {
        match self {
            LossFunction::Squared(loss) => loss.get_unsafe_update(prediction, label, update_scale),
            LossFunction::Logistic(loss) => loss.get_unsafe_update(prediction, label, update_scale),
        }
    }

    fn get_square_grad(&self, prediction: f32, label: f32) -> f32 {
        match self {
            LossFunction::Squared(loss) => loss.get_square_grad(prediction, label),
            LossFunction::Logistic(loss) => loss.get_square_grad(prediction, label),
        }
    }

    fn first_derivative(&self, min_label: f32, max_label: f32, prediction: f32, label: f32) -> f32 {
        match self {
            LossFunction::Squared(loss) => {
                loss.first_derivative(min_label, max_label, prediction, label)
            }
            LossFunction::Logistic(loss) => {
                loss.first_derivative(min_label, max_label, prediction, label)
            }
        }
    }

    fn second_derivative(
        &self,
        min_label: f32,
        max_label: f32,
        prediction: f32,
        label: f32,
    ) -> f32 {
        match self {
            LossFunction::Squared(loss) => {
                loss.second_derivative(min_label, max_label, prediction, label)
            }
            LossFunction::Logistic(loss) => {
                loss.second_derivative(min_label, max_label, prediction, label)
            }
        }
    }
}
