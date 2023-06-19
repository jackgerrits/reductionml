use crate::{
    types::{Label, Prediction},
    Features,
};

pub trait Metric {
    fn add_point(&mut self, features: &Features, label: &Label, prediction: &Prediction);
    fn get_value(&self) -> MetricValue;
    fn get_name(&self) -> String;
}

pub enum MetricValue {
    Bool(bool),
    Float(f32),
    Int(i32),
    String(String),
}

impl ToString for MetricValue {
    fn to_string(&self) -> String {
        match self {
            MetricValue::Bool(b) => b.to_string(),
            MetricValue::Float(f) => f.to_string(),
            MetricValue::Int(i) => i.to_string(),
            MetricValue::String(s) => s.clone(),
        }
    }
}
