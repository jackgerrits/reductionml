use crate::{
    hash::FNV_PRIME,
    sparse_namespaced_features::{constant_feature_index, Namespace, SparseFeatures},
    FeatureHash, FeatureIndex, FeatureMask, ModelIndex,
};

pub trait Weights {
    fn weight_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> f32;
    fn weight_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut f32;

    // By convention state 0 is always the weight itself
    fn state_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> &[f32];
    // By convention state 0 is always the weight itself
    fn state_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut [f32];
}

macro_rules! generate_foreach_feature_func {
    ($func_name: ident, $weight_type: ty, $inner_func_type: ty, $weight_at_func: ident) => {
        pub fn $func_name<F, W>(
            model_offset: ModelIndex,
            features: &SparseFeatures,
            weights: $weight_type,
            quadratic_interactions: &[(Namespace, Namespace)],
            cubic_interactions: &[(Namespace, Namespace, Namespace)],
            num_bits: u8,
            constant_feature_enabled: bool,
            mut func: F,
        ) where
            F: FnMut(f32, $inner_func_type),
            W: Weights,
        {
            for (index, value) in features.all_features() {
                let model_weight = weights.$weight_at_func(index, model_offset);
                func(value, model_weight);
            }

            let masker = FeatureMask::from_num_bits(num_bits);
            // quadratics
            for (ns1, ns2) in quadratic_interactions {
                let same_ns = ns1 == ns2;
                if let Some(ns1) = features.get_namespace(*ns1) {
                    for (i, feat1) in ns1.iter().enumerate() {
                        let multiplied =
                            (FNV_PRIME as u64).wrapping_mul(u32::from(feat1.0) as u64) as u32;
                        if let Some(ns2) = features.get_namespace(*ns2) {
                            for feat2 in ns2.iter().skip(if same_ns { i } else { 0 }) {
                                let idx =
                                    FeatureHash::from(multiplied ^ u32::from(feat2.0)).mask(masker);
                                let model_weight = weights.$weight_at_func(idx, model_offset);
                                func(feat1.1 * feat2.1, model_weight);
                            }
                        }
                    }
                }
            }

            // cubics
            for (ns1, ns2, ns3) in cubic_interactions {
                let same_ns = ns1 == ns2;
                let same_ns2 = ns2 == ns3;
                if let Some(ns1) = features.get_namespace(*ns1) {
                    for (i, feat1) in ns1.iter().enumerate() {
                        let halfhash1 =
                            (FNV_PRIME as u64).wrapping_mul(u32::from(feat1.0) as u64) as u32;
                        if let Some(ns2) = features.get_namespace(*ns2) {
                            for feat2 in ns2.iter().skip(if same_ns { i } else { 0 }) {
                                let halfhash2 = (FNV_PRIME as u64)
                                    .wrapping_mul((halfhash1 ^ u32::from(feat2.0)) as u64)
                                    as u32;
                                if let Some(ns3) = features.get_namespace(*ns3) {
                                    for feat3 in ns3.iter().skip(if same_ns2 { i } else { 0 }) {
                                        let idx = FeatureHash::from(halfhash2 ^ u32::from(feat3.0))
                                            .mask(masker);
                                        let model_weight =
                                            weights.$weight_at_func(idx, model_offset);
                                        func(feat1.1 * feat2.1 * feat3.1, model_weight);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constant_feature_enabled {
                let constant_feature_index = constant_feature_index(num_bits);
                let model_weight = weights.$weight_at_func(constant_feature_index, model_offset);
                func(1.0, model_weight);
            }
        }
    };
}

generate_foreach_feature_func!(foreach_feature, &W, f32, weight_at);
generate_foreach_feature_func!(foreach_feature_with_state, &W, &[f32], state_at);
generate_foreach_feature_func!(
    foreach_feature_with_state_mut,
    &mut W,
    &mut [f32],
    state_at_mut
);
