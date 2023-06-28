use crate::metrics::{ips, parsed_features};

use super::{example_number, mean_squared_error, Metric};

pub fn get_metric(name: &str) -> Option<Box<dyn Metric>> {
    match name {
        "mse" => Some(Box::new(mean_squared_error::MeanSquaredErrorMetric::new())),
        "ips" => Some(Box::new(ips::IpsMetric::new())),
        "parsed_features" => Some(Box::new(parsed_features::ParsedFeaturesMetric::new())),
        "example_number" => Some(Box::new(example_number::ExampleNumberMetric::new())),
        _ => None,
    }
}
