use std::collections::HashMap;

use approx::AbsDiffEq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    error::{Error, Result},
    weights::Weights,
    FeatureIndex, ModelIndex, RawWeightsIndex, StateIndex,
};

fn num_bits_to_represent(val: u64) -> u64 {
    64 - val.leading_zeros() as u64
}

#[derive(Deserialize, Serialize, PartialEq, Debug, Clone)]
pub struct DenseWeights {
    #[serde(
        deserialize_with = "deserialize_sparse_f32_vec",
        serialize_with = "serialize_sparse_f32_vec"
    )]
    weights: Vec<f32>,
    // Max size of index
    feature_index_size: FeatureIndex,
    model_index_size: ModelIndex,
    feature_state_size: StateIndex,
    // Number of bits required to represent index
    model_index_size_shift: u8,
    feature_state_size_shift: u8,
}

impl AbsDiffEq for DenseWeights {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        core::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.feature_index_size != other.feature_index_size
            || self.model_index_size != other.model_index_size
            || self.feature_state_size != other.feature_state_size
            || self.model_index_size_shift != other.model_index_size_shift
            || self.feature_state_size_shift != other.feature_state_size_shift
        {
            return false;
        }

        if self.weights.len() != other.weights.len() {
            return false;
        }

        for i in 0..self.weights.len() {
            if !self.weights[i].abs_diff_eq(&other.weights[i], epsilon) {
                return false;
            }
        }

        true
    }
}

#[derive(Deserialize, Serialize)]
pub struct DenseWeightsWithNDArray {
    weights: HashMap<FeatureIndex, Vec<Vec<f32>>>,
    // Max size of index
    feature_index_size: FeatureIndex,
    model_index_size: ModelIndex,
    feature_state_size: StateIndex,
    // Number of bits required to represent index
    model_index_size_shift: u8,
    feature_state_size_shift: u8,
}

impl DenseWeightsWithNDArray {
    pub fn from_dense_weights(weights: DenseWeights) -> Self {
        let mut weights_map = HashMap::new();
        let feature_index_size = weights.feature_index_size;
        let model_index_size = weights.model_index_size;
        let feature_state_size = weights.feature_state_size;
        let model_index_size_shift = weights.model_index_size_shift;
        let feature_state_size_shift = weights.feature_state_size_shift;

        for i in 0..*feature_index_size {
            let mut found_non_zero = false;
            let mut vec = Vec::new();
            for j in 0..*model_index_size {
                let state = weights.state_at(i.into(), j.into());
                if state.iter().any(|x| *x != 0.0) {
                    found_non_zero = true;
                }
                vec.push(state.into());
            }
            if found_non_zero {
                weights_map.insert(FeatureIndex::from(i), vec);
            }
        }

        DenseWeightsWithNDArray {
            weights: weights_map,
            feature_index_size,
            model_index_size,
            feature_state_size,
            model_index_size_shift,
            feature_state_size_shift,
        }
    }

    pub fn to_dense_weights(&self) -> DenseWeights {
        let feature_index_size_shift =
            num_bits_to_represent(*self.feature_index_size as u64 - 1) as usize;

        let weights = vec![
            0.0;
            (1 << feature_index_size_shift)
                * (1 << self.model_index_size_shift)
                * (1 << self.feature_state_size_shift)
        ];

        let mut weights = DenseWeights {
            weights,
            feature_index_size: self.feature_index_size,
            model_index_size: self.model_index_size,
            feature_state_size: self.feature_state_size,
            model_index_size_shift: self.model_index_size_shift,
            feature_state_size_shift: self.feature_state_size_shift,
        };

        for (feature_index, feature) in &self.weights {
            for (model_index, state) in feature.iter().enumerate() {
                weights
                    .state_at_mut(*feature_index, ModelIndex::from(model_index as u8))
                    .copy_from_slice(state);
            }
        }

        weights
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct SparseF32Vec {
    len: u64,
    non_zero_value_and_index_pairs: Vec<(usize, f32)>,
}

impl SparseF32Vec {
    fn from_dense(vec: &Vec<f32>) -> SparseF32Vec {
        let len: u64 = vec.len().try_into().unwrap();
        let non_zero_value_and_index_pairs: Vec<(usize, f32)> = vec
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, v)| (i, *v))
            .collect();
        SparseF32Vec {
            len,
            non_zero_value_and_index_pairs,
        }
    }

    fn to_dense(&self) -> Vec<f32> {
        let mut vec = vec![0.0; self.len as usize];
        for (index, value) in self.non_zero_value_and_index_pairs.iter() {
            vec[*index] = *value;
        }
        vec
    }
}

fn serialize_sparse_f32_vec<S>(
    vec: &Vec<f32>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let sparse_vec = SparseF32Vec::from_dense(vec);
    sparse_vec.serialize(serializer)
}

fn deserialize_sparse_f32_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let sparse_vec = SparseF32Vec::deserialize(deserializer)?;
    Ok(sparse_vec.to_dense())
}

impl DenseWeights {
    fn convert_index(
        &self,
        feature_index: FeatureIndex,
        model_index: ModelIndex,
    ) -> RawWeightsIndex {
        // These are very good checks but are expensive and in the hot path, so we disable them in release
        debug_assert!(feature_index < self.feature_index_size);
        debug_assert!(model_index < self.model_index_size);
        let raw_index = ((*feature_index as usize)
            << (self.model_index_size_shift + self.feature_state_size_shift))
            + ((*model_index as usize) << self.feature_state_size_shift);

        RawWeightsIndex::from(raw_index)
    }

    pub fn new(
        feature_index_size: FeatureIndex,
        model_index_size: ModelIndex,
        feature_state_size: StateIndex,
    ) -> Result<DenseWeights> {
        let feature_index_size_shift =
            num_bits_to_represent(*feature_index_size as u64 - 1) as usize;
        let model_index_size_shift = num_bits_to_represent(*model_index_size as u64 - 1) as usize;
        let feature_state_size_shift =
            num_bits_to_represent(*feature_state_size as u64 - 1) as usize;
        assert!(feature_index_size_shift + model_index_size_shift + feature_state_size_shift <= 64);
        let weights = vec![
            0.0;
            (1 << feature_index_size_shift)
                * (1 << model_index_size_shift)
                * (1 << feature_state_size_shift)
        ];
        Ok(DenseWeights {
            weights,
            feature_index_size,
            model_index_size,
            feature_state_size,
            // TODO better error message
            model_index_size_shift: u8::try_from(model_index_size_shift)
                .map_err(|e| Error::InvalidArgument(e.to_string()))?,
            feature_state_size_shift: u8::try_from(feature_state_size_shift)
                .map_err(|e| Error::InvalidArgument(e.to_string()))?,
        })
    }
}

impl Weights for DenseWeights {
    fn weight_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> f32 {
        let index = self.convert_index(feature_index, model_index);
        self.weights[*index]
    }

    fn weight_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut f32 {
        let index = self.convert_index(feature_index, model_index);
        &mut self.weights[*index]
    }

    fn state_at(&self, feature_index: FeatureIndex, model_index: ModelIndex) -> &[f32] {
        let index = self.convert_index(feature_index, model_index);
        &self.weights[*index..*index + *self.feature_state_size as usize]
    }

    fn state_at_mut(&mut self, feature_index: FeatureIndex, model_index: ModelIndex) -> &mut [f32] {
        let index = self.convert_index(feature_index, model_index);
        &mut self.weights[*index..*index + *self.feature_state_size as usize]
    }
}

// void foreach_feature(std::uint64_t model_offset, const SparseFeatures& features, const cb::DenseWeights& weights, std::invocable<float, float> auto func)
// {
//   for (const auto[index, value] : features.flat_values_and_indices())
//   {
//     const auto model_weight = weights.weight_at(index, model_offset);
//     func(value, model_weight);
//   }
// }

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};

    use super::*;

    #[test]
    fn test_num_bits_to_represent() {
        assert_eq!(num_bits_to_represent(0), 0);
        assert_eq!(num_bits_to_represent(1), 1);
        assert_eq!(num_bits_to_represent(2), 2);
        assert_eq!(num_bits_to_represent(3), 2);
        assert_eq!(num_bits_to_represent(4), 3);
        assert_eq!(num_bits_to_represent(5), 3);
        assert_eq!(num_bits_to_represent(6), 3);
        assert_eq!(num_bits_to_represent(7), 3);
        assert_eq!(num_bits_to_represent(8), 4);
        assert_eq!(num_bits_to_represent(9), 4);
        assert_eq!(num_bits_to_represent(10), 4);
        assert_eq!(num_bits_to_represent(11), 4);
        assert_eq!(num_bits_to_represent(12), 4);
        assert_eq!(num_bits_to_represent(13), 4);
        assert_eq!(num_bits_to_represent(14), 4);
        assert_eq!(num_bits_to_represent(15), 4);
        assert_eq!(num_bits_to_represent(16), 5);
        assert_eq!(num_bits_to_represent(17), 5);
        assert_eq!(num_bits_to_represent(18), 5);
        assert_eq!(num_bits_to_represent(19), 5);
        assert_eq!(num_bits_to_represent(20), 5);
        assert_eq!(num_bits_to_represent(21), 5);
        assert_eq!(num_bits_to_represent(22), 5);
        assert_eq!(num_bits_to_represent(23), 5);
        assert_eq!(num_bits_to_represent(24), 5);
        assert_eq!(num_bits_to_represent(25), 5);
        assert_eq!(num_bits_to_represent(26), 5);
        assert_eq!(num_bits_to_represent(27), 5);
        assert_eq!(num_bits_to_represent(28), 5);
        assert_eq!(num_bits_to_represent(29), 5);
        assert_eq!(num_bits_to_represent(30), 5);
        assert_eq!(num_bits_to_represent(31), 5);
    }

    #[test]
    fn weights_equality() {
        let mut w1 = DenseWeights::new(
            FeatureIndex::from(4),
            ModelIndex::from(1),
            StateIndex::from(1),
        )
        .unwrap();
        let w2 = DenseWeights::new(
            FeatureIndex::from(4),
            ModelIndex::from(1),
            StateIndex::from(1),
        )
        .unwrap();

        assert_abs_diff_eq!(w1, w2);

        *w1.weight_at_mut(FeatureIndex::from(0), ModelIndex::from(0)) = 1.0;

        assert_abs_diff_ne!(w1, w2);
    }

    #[test]
    fn weights_roundtrip() {
        let mut w1 = DenseWeights::new(
            FeatureIndex::from(4),
            ModelIndex::from(2),
            StateIndex::from(3),
        )
        .unwrap();
        for i in 0..4 {
            *w1.weight_at_mut(FeatureIndex::from(i), ModelIndex::from(0)) = i as f32;
            *w1.weight_at_mut(FeatureIndex::from(i), ModelIndex::from(1)) = i as f32 * 2_f32;
            w1.state_at_mut(FeatureIndex::from(i), ModelIndex::from(0))[1] = i as f32 * 3_f32;
            w1.state_at_mut(FeatureIndex::from(i), ModelIndex::from(1))[2] = i as f32 * 3_f32;
        }

        let dwnd = DenseWeightsWithNDArray::from_dense_weights(w1.clone());

        let w2 = dwnd.to_dense_weights();

        assert_abs_diff_eq!(w1, w2);
    }
}
