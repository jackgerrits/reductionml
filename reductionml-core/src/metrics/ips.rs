use crate::{metrics::Metric, utils::AsInner, ActionProbsPrediction, CBLabel, Features};

use super::MetricValue;

pub struct IpsMetric {
    pub examples_count: u64,
    pub weighted_reward: f32,
}

impl IpsMetric {
    pub fn new() -> IpsMetric {
        IpsMetric {
            examples_count: 0,
            weighted_reward: 0.0,
        }
    }
}

impl Default for IpsMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for IpsMetric {
    fn add_point(
        &mut self,
        _features: &Features,
        label: &crate::types::Label,
        prediction: &crate::types::Prediction,
    ) {
        let label: &CBLabel = label.as_inner().unwrap();
        let pred: &ActionProbsPrediction = prediction.as_inner().unwrap();

        let p_log = label.probability;
        let p_pred = pred
            .0
            .iter()
            .find(|(action, _)| action == &label.action)
            .unwrap()
            .1;

        let w = p_pred / p_log;

        self.weighted_reward += (-1.0 * label.cost) * w;
        self.examples_count += 1;
    }

    fn get_value(&self) -> MetricValue {
        MetricValue::Float(self.weighted_reward / (self.examples_count as f32))
    }

    fn get_name(&self) -> String {
        "Estimated reward (IPS)".to_owned()
    }
}
