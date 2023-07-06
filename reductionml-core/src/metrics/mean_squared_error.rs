use crate::{metrics::Metric, utils::AsInner, Features, ScalarPrediction, SimpleLabel};

use super::MetricValue;

pub struct MeanSquaredErrorMetric {
    pub value: f32,
    pub count: u64,
}

impl MeanSquaredErrorMetric {
    pub fn new() -> MeanSquaredErrorMetric {
        MeanSquaredErrorMetric {
            value: 0.0,
            count: 0,
        }
    }
}

impl Default for MeanSquaredErrorMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MeanSquaredErrorMetric {
    fn add_point(
        &mut self,
        _features: &Features,
        label: &crate::types::Label,
        prediction: &crate::types::Prediction,
    ) {
        let label: &SimpleLabel = label.as_inner().unwrap();
        let pred: &ScalarPrediction = prediction.as_inner().unwrap();
        self.value += (label.value() - pred.prediction) * (label.value() - pred.prediction);
        self.count += 1;
    }

    fn get_value(&self) -> MetricValue {
        MetricValue::Float(self.value / self.count as f32)
    }

    fn get_name(&self) -> String {
        "MeanSquaredError".to_owned()
    }
}
