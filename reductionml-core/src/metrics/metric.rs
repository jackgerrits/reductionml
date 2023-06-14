use crate::{
    types::{Label, Prediction}
};

pub trait Metric {
    fn add_point(&mut self, label: &Label, prediction: &Prediction);
    fn get_value(&self) -> f32;
    fn get_name(&self) -> String;
}
