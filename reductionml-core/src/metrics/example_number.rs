use crate::{metrics::Metric, Features};

use super::MetricValue;

pub struct ExampleNumberMetric {
    pub count: u64,
}

impl ExampleNumberMetric {
    pub fn new() -> ExampleNumberMetric {
        ExampleNumberMetric { count: 0 }
    }
}

impl Default for ExampleNumberMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for ExampleNumberMetric {
    fn add_point(
        &mut self,
        _features: &Features,
        _label: &crate::types::Label,
        _prediction: &crate::types::Prediction,
    ) {
        self.count += 1;
    }

    fn get_value(&self) -> MetricValue {
        if self.count == 0 {
            panic!("Cannot get value of ExampleNumberMetric with no points");
        }
        MetricValue::Int(self.count as i32 - 1)
    }

    fn get_name(&self) -> String {
        "Example #".to_owned()
    }
}
