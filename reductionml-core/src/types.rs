use serde::{Deserialize, Serialize};

use crate::{
    object_pool::PoolReturnable, sparse_namespaced_features::SparseFeatures, utils::GetInner,
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

        impl GetInner<$structname> for $target_type {
            fn get_inner_ref(&self) -> Option<&$structname> {
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
#[derive(Clone, Copy, Debug)]
pub struct SimpleLabel(pub f32, pub f32);

impl From<f32> for SimpleLabel {
    fn from(f: f32) -> Self {
        SimpleLabel(f, 1.0)
    }
}

impl Default for SimpleLabel {
    fn default() -> Self {
        SimpleLabel(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryLabel(pub bool);
impl From<bool> for BinaryLabel {
    fn from(b: bool) -> Self {
        BinaryLabel(b)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CBLabel {
    // action is 0-based
    pub action: usize,
    pub cost: f32,
    pub probability: f32,
}

#[derive(Clone, Copy, Debug)]
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

#[derive(PartialEq, Clone, Debug)]
pub struct CBAdfFeatures {
    pub shared: Option<SparseFeatures>,
    pub actions: Vec<SparseFeatures>,
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

        impl<'a> From<&'a $structname> for Features<'a> {
            fn from(f: &'a $structname) -> Features<'a> {
                Features::$enum_variant_ref(f)
            }
        }

        impl GetInner<$structname> for Features<'_> {
            fn get_inner_ref(&self) -> Option<&$structname> {
                match self {
                    Features::$enum_variant(f) => Some(f),
                    Features::$enum_variant_ref(f) => Some(f),
                    _ => None,
                }
            }
        }
    };
}

#[derive(PartialEq, Clone, Debug)]
pub enum Features<'a> {
    SparseSimple(SparseFeatures),
    SparseSimpleRef(&'a SparseFeatures),
    SparseCBAdf(CBAdfFeatures),
    SparseCBAdfRef(&'a CBAdfFeatures),
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, Debug)]
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
