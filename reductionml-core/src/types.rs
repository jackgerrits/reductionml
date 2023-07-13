use approx::AbsDiffEq;
use serde::{Deserialize, Serialize};

use crate::{
    object_pool::PoolReturnable, sparse_namespaced_features::SparseFeatures, utils::AsInner,
};
use derive_more::TryInto;
use std::ops::Deref;
macro_rules! impl_conversion_traits {
    ($target_type: ident, $enum_variant: ident, $structname: ident) => {
        impl From<$structname> for $target_type {
            fn from(f: $structname) -> Self {
                $target_type::$enum_variant(f)
            }
        }

        impl AsInner<$structname> for $target_type {
            fn as_inner(&self) -> Option<&$structname> {
                match self {
                    $target_type::$enum_variant(f) => Some(f),
                    _ => None,
                }
            }
            fn as_inner_mut(&mut self) -> Option<&mut $structname> {
                match self {
                    $target_type::$enum_variant(f) => Some(f),
                    _ => None,
                }
            }
        }
    };
}

#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct ScalarPrediction {
    pub prediction: f32,
    pub raw_prediction: f32,
}

#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct BinaryPrediction(pub bool);
impl From<bool> for BinaryPrediction {
    fn from(b: bool) -> Self {
        BinaryPrediction(b)
    }
}

#[derive(Debug, PartialEq, Clone, Default, Serialize)]
pub struct ActionScoresPrediction(pub Vec<(usize, f32)>);

#[derive(Debug, PartialEq, Clone, Default, Serialize)]
pub struct ActionProbsPrediction(pub Vec<(usize, f32)>);

#[derive(Debug, PartialEq, Clone, TryInto, Serialize)]
// Untagged for succintness in predictions files
#[serde(untagged)]
pub enum Prediction {
    Scalar(ScalarPrediction),
    Binary(BinaryPrediction),
    ActionScores(ActionScoresPrediction),
    ActionProbs(ActionProbsPrediction),
}

impl_conversion_traits!(Prediction, Scalar, ScalarPrediction);
impl_conversion_traits!(Prediction, Binary, BinaryPrediction);
impl_conversion_traits!(Prediction, ActionScores, ActionScoresPrediction);
impl_conversion_traits!(Prediction, ActionProbs, ActionProbsPrediction);

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum PredictionType {
    Scalar,
    Binary,
    ActionScores,
    ActionProbs,
}

/// value, weight
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SimpleLabel {
    value: f32,
    weight: f32,
}

impl SimpleLabel {
    pub fn new(value: f32, weight: f32) -> Self {
        SimpleLabel { value, weight }
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }
}

impl From<f32> for SimpleLabel {
    fn from(f: f32) -> Self {
        SimpleLabel::new(f, 1.0)
    }
}

impl Default for SimpleLabel {
    fn default() -> Self {
        SimpleLabel::new(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BinaryLabel(pub bool);

impl From<bool> for BinaryLabel {
    fn from(b: bool) -> Self {
        BinaryLabel(b)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CBLabel {
    // action is 0-based
    pub action: usize,
    pub cost: f32,
    pub probability: f32,
}

impl CBLabel {
    pub fn new(action: usize, cost: f32, probability: f32) -> Self {
        CBLabel {
            action,
            cost,
            probability,
        }
    }

    pub fn action(&self) -> usize {
        self.action
    }

    pub fn cost(&self) -> f32 {
        self.cost
    }

    pub fn probability(&self) -> f32 {
        self.probability
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Label {
    Simple(SimpleLabel),
    Binary(BinaryLabel),
    CB(CBLabel),
}
impl_conversion_traits!(Label, Simple, SimpleLabel);
impl_conversion_traits!(Label, Binary, BinaryLabel);
impl_conversion_traits!(Label, CB, CBLabel);

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum LabelType {
    Simple,
    Binary,
    CB,
}

#[derive(PartialEq, Clone, Debug, Default)]
pub struct CBAdfFeatures {
    pub shared: Option<SparseFeatures>,
    pub actions: Vec<SparseFeatures>,
}

impl AbsDiffEq for CBAdfFeatures {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        core::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if let (Some(shared), Some(other_shared)) = (&self.shared, &other.shared) {
            if !shared.abs_diff_eq(other_shared, epsilon) {
                return false;
            }
        } else if let (None, None) = (&self.shared, &other.shared) {
            // ok
        } else {
            return false;
        }

        if self.actions.len() != other.actions.len() {
            return false;
        }
        for (a, b) in self.actions.iter().zip(other.actions.iter()) {
            if !a.abs_diff_eq(b, epsilon) {
                return false;
            }
        }
        true
    }
}

impl PoolReturnable<SparseFeatures> for CBAdfFeatures {
    fn clear_and_return_object(self, pool: &crate::object_pool::Pool<SparseFeatures>) {
        if let Some(shared) = self.shared {
            shared.clear_and_return_object(pool);
        }
        for action in self.actions {
            action.clear_and_return_object(pool);
        }
    }
}

macro_rules! impl_conversion_traits_feats {
    ($enum_variant: ident, $enum_variant_ref: ident, $structname: ident) => {
        impl From<$structname> for Features<'_> {
            fn from(f: $structname) -> Self {
                Features::$enum_variant(f)
            }
        }

        // impl From<&$structname> for Features<'_> {
        //     fn from(f: &$structname) -> Self {
        //         Features::$enum_variant_ref(f)
        //     }
        // }

        impl<'a> From<&'a mut $structname> for Features<'a> {
            fn from(f: &'a mut $structname) -> Features<'a> {
                Features::$enum_variant_ref(f)
            }
        }

        impl AsInner<$structname> for Features<'_> {
            fn as_inner(&self) -> Option<&$structname> {
                match self {
                    Features::$enum_variant(f) => Some(f),
                    Features::$enum_variant_ref(f) => Some(f),
                    _ => None,
                }
            }

            fn as_inner_mut(&mut self) -> Option<&mut $structname> {
                match self {
                    Features::$enum_variant(f) => Some(f),
                    Features::$enum_variant_ref(f) => Some(f),
                    _ => None,
                }
            }
        }
    };
}

#[derive(PartialEq, Debug)]
pub enum Features<'a> {
    SparseSimple(SparseFeatures),
    SparseSimpleRef(&'a mut SparseFeatures),
    SparseCBAdf(CBAdfFeatures),
    SparseCBAdfRef(&'a mut CBAdfFeatures),
}

impl<'a> Features<'a> {
    pub fn clone(&self) -> Features<'static> {
        match self {
            Features::SparseSimple(f) => Features::SparseSimple(f.clone()),
            Features::SparseSimpleRef(f) => Features::SparseSimple((*f).clone()),
            Features::SparseCBAdf(f) => Features::SparseCBAdf(f.clone()),
            Features::SparseCBAdfRef(f) => Features::SparseCBAdf((*f).clone()),
        }
    }
}

impl<'a> AbsDiffEq for Features<'a> {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        core::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        match (self, other) {
            (
                Features::SparseSimple(_) | Features::SparseSimpleRef(_),
                Features::SparseSimple(_) | Features::SparseSimpleRef(_),
            ) => {
                let left: &SparseFeatures = self.as_inner().unwrap();
                let right: &SparseFeatures = other.as_inner().unwrap();
                left.abs_diff_eq(right, epsilon)
            }
            (
                Features::SparseCBAdf(_) | Features::SparseCBAdfRef(_),
                Features::SparseCBAdf(_) | Features::SparseCBAdfRef(_),
            ) => {
                let left: &CBAdfFeatures = self.as_inner().unwrap();
                let right: &CBAdfFeatures = other.as_inner().unwrap();
                left.abs_diff_eq(right, epsilon)
            }
            (_, _) => false,
        }
    }
}

// impl_conversion_traits!(Features, SparseSimple, SparseFeatures);
// impl_conversion_traits!(Features, SparseCBAdf, CBAdfFeatures);

impl_conversion_traits_feats!(SparseSimple, SparseSimpleRef, SparseFeatures);
impl_conversion_traits_feats!(SparseCBAdf, SparseCBAdfRef, CBAdfFeatures);

// impl From<SparseFeatures> for Features<'_> {
//     fn from(f: SparseFeatures) -> Self {
//         Features::SparseSimple(f)
//     }
// }

// impl<'a> From<&'a SparseFeatures> for Features<'a> {
//     fn from(f: &'a SparseFeatures) -> Features<'a> {
//         Features::SparseSimpleRef(f)
//     }
// }

// impl GetInner<SparseFeatures> for Features<'_> {
//     fn get_inner_ref(&self) -> Option<&SparseFeatures> {
//         match self {
//             Features::SparseSimple(f) => Some(f),
//             Features::SparseSimpleRef(f) => Some(f),
//             _ => None,
//         }
//     }
// }

impl TryFrom<Features<'_>> for SparseFeatures {
    type Error = &'static str;

    fn try_from(value: Features) -> Result<Self, Self::Error> {
        match value {
            Features::SparseSimple(f) => Ok(f),
            _ => Err("Cannot convert to SparseFeatures"),
        }
    }
}

impl TryFrom<Features<'_>> for CBAdfFeatures {
    type Error = &'static str;

    fn try_from(value: Features) -> Result<Self, Self::Error> {
        match value {
            Features::SparseCBAdf(f) => Ok(f),
            _ => Err("Cannot convert to CBAdfFeatures"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum FeaturesType {
    SparseSimple,
    SparseCBAdf,
}

impl PoolReturnable<SparseFeatures> for Features<'_> {
    fn clear_and_return_object(self, pool: &crate::object_pool::Pool<SparseFeatures>) {
        match self {
            Features::SparseSimple(obj) => {
                obj.clear_and_return_object(pool);
            }
            Features::SparseSimpleRef(_) => (),
            Features::SparseCBAdf(obj) => obj.clear_and_return_object(pool),
            Features::SparseCBAdfRef(_) => (),
        }
    }
}

// Index types

macro_rules! impl_extra_traits {
    ($structname: ident, $inner_type: ident ) => {
        impl From<$inner_type> for $structname {
            fn from(value: $inner_type) -> Self {
                $structname(value)
            }
        }

        impl Deref for $structname {
            type Target = $inner_type;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl From<$structname> for $inner_type {
            fn from(value: $structname) -> Self {
                value.0
            }
        }
    };
}

/// This type must be masked by the num bits in use in the hash table.
#[derive(Deserialize, Serialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct FeatureIndex(u32);
impl_extra_traits!(FeatureIndex, u32);

#[derive(Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct ModelIndex(u8);
impl_extra_traits!(ModelIndex, u8);

#[derive(Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct StateIndex(u8);
impl_extra_traits!(StateIndex, u8);

#[derive(Clone, Copy)]
pub struct FeatureMask(u32);
impl_extra_traits!(FeatureMask, u32);

impl FeatureMask {
    pub fn from_num_bits(num_bits: u8) -> FeatureMask {
        FeatureMask((1_u32 << num_bits) - 1)
    }

    pub fn mask(&self, index: FeatureHash) -> FeatureIndex {
        FeatureIndex(index.0 & self.0)
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Deserialize, Serialize, Debug)]
pub struct NamespaceHash(u32);
impl_extra_traits!(NamespaceHash, u32);

// This is the full 32 bit hash of the feature.
#[derive(Clone, Copy)]
pub struct FeatureHash(u32);
impl_extra_traits!(FeatureHash, u32);

impl FeatureHash {
    pub fn mask(&self, mask: FeatureMask) -> FeatureIndex {
        mask.mask(*self)
    }
}

/// Intentionally internal only, used in dense weights
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) struct RawWeightsIndex(usize);
impl_extra_traits!(RawWeightsIndex, usize);
