use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::{
    reduction_factory::ReductionFactory,
    reductions::{
        BinaryReductionFactory, CBAdfReductionFactory, CBExploreAdfGreedyReductionFactory,
        CBExploreAdfSquareCBReductionFactory, CoinRegressorFactory, DebugReductionFactory,
    },
};

pub static REDUCTION_REGISTRY: Lazy<RwLock<ReductionRegistry>> = Lazy::new(|| {
    let mut registry: ReductionRegistry = ReductionRegistry::default();
    registry.register(Box::<CoinRegressorFactory>::default());
    registry.register(Box::<BinaryReductionFactory>::default());
    registry.register(Box::<CBAdfReductionFactory>::default());
    registry.register(Box::<CBExploreAdfGreedyReductionFactory>::default());
    registry.register(Box::<DebugReductionFactory>::default());
    registry.register(Box::<CBExploreAdfSquareCBReductionFactory>::default());
    RwLock::new(registry)
});

#[derive(Default)]
pub struct ReductionRegistry {
    registry: BTreeMap<String, Box<dyn ReductionFactory>>,
}

// impl Send for ReductionRegistry {}
unsafe impl Sync for ReductionRegistry {}
unsafe impl Send for ReductionRegistry {}

impl ReductionRegistry {
    pub fn register(&mut self, factory: Box<dyn ReductionFactory>) {
        self.registry
            .insert(factory.typename().as_ref().to_owned(), factory);
    }

    pub fn get(&self, typename: &str) -> Option<&dyn ReductionFactory> {
        self.registry.get(typename).map(|x| x.as_ref())
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn ReductionFactory> {
        self.registry.values().map(|x| x.as_ref())
    }
}
