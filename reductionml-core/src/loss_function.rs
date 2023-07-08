mod squared_loss;

use serde::{Deserialize, Serialize};

use self::squared_loss::SquaredLoss;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LossFunctionType {
    Squared,
}

impl LossFunctionType {
    pub fn create(&self) -> Box<dyn LossFunction> {
        match self {
            LossFunctionType::Squared => Box::new(SquaredLoss::new()),
        }
    }
}

pub trait LossFunction: Send {
    fn get_type(&self) -> LossFunctionType;
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
