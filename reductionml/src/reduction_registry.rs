use std::collections::HashMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::{
    reduction_factory::ReductionFactory,
    reductions::{BinaryReductionFactory, CBAdfReductionFactory, CoinRegressorFactory},
};

pub static REDUCTION_REGISTRY: Lazy<RwLock<ReductionRegistry>> = Lazy::new(|| {
    let mut registry: ReductionRegistry = ReductionRegistry::default();
    registry.register(Box::<CoinRegressorFactory>::default());
    registry.register(Box::<BinaryReductionFactory>::default());
    registry.register(Box::<CBAdfReductionFactory>::default());
    RwLock::new(registry)
});

#[derive(Default)]
pub struct ReductionRegistry {
    registry: HashMap<String, Box<dyn ReductionFactory>>,
}

// impl Send for ReductionRegistry {}
unsafe impl Sync for ReductionRegistry {}
unsafe impl Send for ReductionRegistry {}

impl ReductionRegistry {
    pub fn register(&mut self, factory: Box<dyn ReductionFactory>) {
        self.registry.insert(factory.typename(), factory);
    }

    pub fn get(&self, typename: &str) -> Option<&dyn ReductionFactory> {
        self.registry.get(typename).map(|x| x.as_ref())
    }
}
