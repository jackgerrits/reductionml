use crate::FeatureIndex;

#[must_use]
pub fn bits_to_max_feature_index(val: u8) -> FeatureIndex {
    FeatureIndex::from(1_u32 << (val as u32))
}

pub trait AsInner<T>: Sized {
    fn as_inner(&self) -> Option<&T>;
    fn as_inner_mut(&mut self) -> Option<&mut T>;
}
