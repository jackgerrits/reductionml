use crate::FeatureIndex;

#[must_use]
pub fn bits_to_max_feature_index(val: u8) -> FeatureIndex {
    FeatureIndex::from(1_u32 << (val as u32))
}

pub trait GetInner<T>: Sized {
    fn get_inner_ref(&self) -> Option<&T>;
}
