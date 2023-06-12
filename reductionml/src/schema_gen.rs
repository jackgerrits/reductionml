use reductionml::{reduction_registry::REDUCTION_REGISTRY, config_schema::ConfigSchema};

fn main() {
    let mut schema = ConfigSchema::new();
    REDUCTION_REGISTRY.read().as_ref().as_ref().unwrap().iter().for_each(|x| {
        schema.add_reduction(x);
    });

    println!("{}", serde_json::to_string_pretty(schema.schema()).unwrap());
}