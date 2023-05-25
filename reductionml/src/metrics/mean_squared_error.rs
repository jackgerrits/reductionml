use crate::{metrics::Metric, utils::GetInner, ScalarPrediction, SimpleLabel};

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

impl Metric for MeanSquaredErrorMetric {
    fn add_point(&mut self, label: &crate::types::Label, prediction: &crate::types::Prediction) {
        let label: &SimpleLabel = label.get_inner_ref().unwrap();
        let pred: &ScalarPrediction = prediction.get_inner_ref().unwrap();
        self.value += (label.0 - pred.prediction) * (label.0 - pred.prediction);
        self.count += 1;
    }

    fn get_value(&self) -> f32 {
        self.value / self.count as f32
    }

    fn get_name(&self) -> String {
        "MeanSquaredError".to_owned()
    }
}
