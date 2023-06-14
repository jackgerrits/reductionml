use crate::{sparse_namespaced_features::SparseFeatures, FeatureIndex, ModelIndex};

pub trait Weights {
    fn weight_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> f32;
    fn weight_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut f32;

    // By convention state 0 is always the weight itself
    fn state_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> &[f32];
    // By convention state 0 is always the weight itself
    fn state_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut [f32];
}

pub fn foreach_feature<F, W>(
    model_offset: ModelIndex,
    features: &SparseFeatures,
    weights: &W,
    mut func: F,
) where
    F: FnMut(f32, f32),
    W: Weights,
{
    for (index, value) in features.all_features() {
        let model_weight = weights.weight_at(index, model_offset);
        func(value, model_weight);
    }
}

pub fn foreach_feature_with_state_mut<F, W>(
    model_offset: ModelIndex,
    features: &SparseFeatures,
    weights: &mut W,
    mut func: F,
) where
    F: FnMut(f32, &mut [f32]),
    W: Weights,
{
    for (index, value) in features.all_features() {
        let model_weight = weights.state_at_mut(index, model_offset);
        func(value, model_weight);
    }
}

pub fn foreach_feature_with_state<F, W>(
    model_offset: ModelIndex,
    features: &SparseFeatures,
    weights: &W,
    mut func: F,
) where
    F: FnMut(f32, &[f32]),
    W: Weights,
{
    for (index, value) in features.all_features() {
        let model_weight = weights.state_at(index, model_offset);
        func(value, model_weight);
    }
}