use std::collections::BTreeSet;

use crate::{
    hash::{hash_bytes, FNV_PRIME},
    object_pool::PoolReturnable,
    utils::bits_to_max_feature_index,
    FeatureHash, FeatureIndex, FeatureMask, NamespaceHash,
};
use approx::AbsDiffEq;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
pub struct NamespacesIterator<'a> {
    namespaces: std::collections::hash_map::Iter<'a, Namespace, SparseFeaturesNamespace>,
}

#[derive(Clone)]
pub struct NamespaceIterator<'a> {
    indices: std::slice::Iter<'a, FeatureIndex>,
    values: std::slice::Iter<'a, f32>,
}

// TODO this should be skipping inactive namespaces.
impl<'a> Iterator for NamespacesIterator<'a> {
    type Item = (Namespace, NamespaceIterator<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        self.namespaces.next().map(|(namespace_feats, namespace)| {
            (
                *namespace_feats,
                NamespaceIterator {
                    indices: namespace.feature_indices.iter(),
                    values: namespace.feature_values.iter(),
                },
            )
        })
    }
}

impl<'a> Iterator for NamespaceIterator<'a> {
    type Item = (FeatureIndex, f32);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.indices.next() {
            Some((*index, *self.values.next().expect(
                "NamespaceIterator::indices and NamespaceIterator::values are not the same length",
            )))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match (self.indices.nth(n), self.values.nth(n)) {
            (Some(i), Some(v)) => Some((*i, *v)),
            _ => None,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct SparseFeaturesNamespace {
    namespace: Namespace,
    feature_indices: Vec<FeatureIndex>,
    feature_values: Vec<f32>,
    /// active is an optimization for usage solely in the SparseFeatures struct
    /// it is used to avoid iterating over the namespace when it is not active
    /// and also allow for object reuse
    active: bool,
}

impl AbsDiffEq for SparseFeaturesNamespace {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        core::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.namespace == other.namespace
            && self.feature_indices == other.feature_indices
            && self
                .feature_values
                .iter()
                .zip(other.feature_values.iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl SparseFeaturesNamespace {
    pub fn iter(&self) -> NamespaceIterator {
        NamespaceIterator {
            indices: self.feature_indices.iter(),
            values: self.feature_values.iter(),
        }
    }

    pub fn new(namespace: Namespace) -> SparseFeaturesNamespace {
        SparseFeaturesNamespace {
            namespace,
            feature_indices: Vec::new(),
            feature_values: Vec::new(),
            active: false,
        }
    }

    pub fn new_with_capacity(namespace: Namespace, capacity: usize) -> SparseFeaturesNamespace {
        SparseFeaturesNamespace {
            namespace,
            feature_indices: Vec::with_capacity(capacity),
            feature_values: Vec::with_capacity(capacity),
            active: false,
        }
    }

    pub fn size(&self) -> usize {
        self.feature_indices.len()
    }

    pub fn namespace(&self) -> Namespace {
        self.namespace
    }

    pub fn reserve(&mut self, size: usize) {
        self.feature_indices
            .reserve_exact(size - self.feature_indices.capacity());
        self.feature_values
            .reserve(size - self.feature_values.capacity());
    }

    pub fn add_feature(&mut self, feature_index: FeatureIndex, feature_value: f32) {
        self.feature_indices.push(feature_index);
        self.feature_values.push(feature_value);
    }

    pub fn add_features(&mut self, feature_indices: &[FeatureIndex], feature_values: &[f32]) {
        assert_eq!(feature_indices.len(), feature_values.len());
        self.feature_indices.extend_from_slice(feature_indices);
        self.feature_values.extend_from_slice(feature_values);
    }

    pub fn add_features_with_iter<I1, I2>(&mut self, feature_indices: I1, feature_values: I2)
    where
        I1: Iterator<Item = FeatureIndex>,
        I2: Iterator<Item = f32>,
    {
        self.feature_indices.extend(feature_indices);
        self.feature_values.extend(feature_values);
        assert_eq!(self.feature_indices.len(), self.feature_values.len());
    }

    fn clear(&mut self) {
        self.feature_indices.clear();
        self.feature_values.clear();
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

#[derive(Serialize, Deserialize, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Namespace {
    Named(NamespaceHash),
    Default,
}

impl Namespace {
    pub fn from_name(namespace_name: &str, hash_seed: u32) -> Namespace {
        match namespace_name {
            // TODO: consider different hash if hash_seed is not 0
            " " => Namespace::Default,
            ":default" => Namespace::Default,
            _ => {
                let namespace_hash = hash_bytes(namespace_name.as_bytes(), hash_seed).into();
                Namespace::Named(namespace_hash)
            }
        }
    }

    pub fn hash(&self, _hash_seed: u32) -> NamespaceHash {
        match self {
            Namespace::Named(hash) => *hash,
            Namespace::Default => 0.into(),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct SparseFeatures {
    namespaces: std::collections::HashMap<Namespace, SparseFeaturesNamespace>,
}

impl Default for SparseFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl AbsDiffEq for SparseFeatures {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        core::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let left_ns: BTreeSet<Namespace> = self
            .namespaces
            .iter()
            .filter_map(|(ns, ns_vals)| if ns_vals.is_active() { Some(ns) } else { None })
            .cloned()
            .collect();

        let right_ns = other
            .namespaces
            .iter()
            .filter_map(|(ns, ns_vals)| if ns_vals.is_active() { Some(ns) } else { None })
            .cloned()
            .collect();
        if left_ns != right_ns {
            return false;
        }

        for ns in left_ns {
            let ns_vals = self.namespaces.get(&ns).unwrap();
            let other_ns_vals = other.namespaces.get(&ns).unwrap();
            if !ns_vals.abs_diff_eq(other_ns_vals, epsilon) {
                return false;
            }
        }

        true
    }
}
// Essentially Fowler–Noll–Vo hash function
fn quadratic_feature_hash(i1: FeatureIndex, i2: FeatureIndex) -> FeatureHash {
    let multiplied = (FNV_PRIME as u64).wrapping_mul(u32::from(i1) as u64) as u32;
    (multiplied ^ u32::from(i2)).into()
}

// Essentially Fowler–Noll–Vo hash function
fn cubic_feature_hash(i1: FeatureIndex, i2: FeatureIndex, i3: FeatureIndex) -> FeatureHash {
    let multiplied = (FNV_PRIME as u64).wrapping_mul(u32::from(i1) as u64) as u32;
    let multiplied = (FNV_PRIME as u64).wrapping_mul((multiplied ^ u32::from(i2)) as u64) as u32;
    (multiplied ^ u32::from(i3)).into()
}

fn feature_space_median_index(num_bits: u8) -> FeatureIndex {
    (u32::from(bits_to_max_feature_index(num_bits)) / 2).into()
}

/// Constant feature is defined to have the median index of the feature space.
pub fn constant_feature_index(num_bits: u8) -> FeatureIndex {
    feature_space_median_index(num_bits)
}

impl SparseFeatures {
    pub fn namespaces(&self) -> NamespacesIterator {
        NamespacesIterator {
            namespaces: self.namespaces.iter(),
        }
    }

    pub fn quadratic_features(
        &self,
        ns1: Namespace,
        ns2: Namespace,
        num_bits: u8,
    ) -> Option<impl Iterator<Item = (FeatureIndex, f32)> + '_> {
        let ns1 = self.get_namespace(ns1)?;
        let ns2 = self.get_namespace(ns2)?;

        let masker = FeatureMask::from_num_bits(num_bits);

        Some(
            ns1.iter()
                .cartesian_product(ns2.iter().clone())
                .map(move |((i1, v1), (i2, v2))| {
                    (quadratic_feature_hash(i1, i2).mask(masker), v1 * v2)
                }),
        )
    }

    pub fn cubic_features(
        &self,
        ns1: Namespace,
        ns2: Namespace,
        ns3: Namespace,
        num_bits: u8,
    ) -> Option<impl Iterator<Item = (FeatureIndex, f32)> + '_> {
        let ns1 = self.get_namespace(ns1)?;
        let ns2 = self.get_namespace(ns2)?;
        let ns3 = self.get_namespace(ns3)?;

        let masker = FeatureMask::from_num_bits(num_bits);

        Some(
            ns1.iter()
                .cartesian_product(ns2.iter().clone())
                .cartesian_product(ns3.iter().clone())
                .map(move |(((i1, v1), (i2, v2)), (i3, v3))| {
                    (cubic_feature_hash(i1, i2, i3).mask(masker), v1 * v2 * v3)
                }),
        )
    }

    pub fn all_features(&self) -> impl Iterator<Item = (FeatureIndex, f32)> + '_ {
        self.namespaces
            .iter()
            .flat_map(|(_, namespace)| namespace.iter())
    }

    pub fn new() -> SparseFeatures {
        SparseFeatures {
            namespaces: std::collections::HashMap::new(),
        }
    }

    pub fn get_namespace(&self, namespace: Namespace) -> Option<&SparseFeaturesNamespace> {
        self.namespaces
            .get(&namespace)
            .filter(|namespace| namespace.is_active())
    }

    pub fn get_namespace_mut(
        &mut self,
        namespace: Namespace,
    ) -> Option<&mut SparseFeaturesNamespace> {
        self.namespaces
            .get_mut(&namespace)
            .filter(|namespace| namespace.is_active())
    }

    pub fn clear(&mut self) {
        for namespace in self.namespaces.values_mut() {
            namespace.clear();
            namespace.set_active(false);
        }
    }

    // pub fn shrink(&mut self) {
    //     for namespace in self.namespaces.values_mut() {
    //         namespace.feature_indices.shrink_to_fit();
    //         namespace.feature_values.shrink_to_fit();
    //     }
    // }

    pub fn get_or_create_namespace(
        &mut self,
        namespace: Namespace,
    ) -> &mut SparseFeaturesNamespace {
        let item = self
            .namespaces
            .entry(namespace)
            .or_insert(SparseFeaturesNamespace::new(namespace));
        item.set_active(true);
        item
    }

    pub fn get_or_create_namespace_with_capacity(
        &mut self,
        namespace: Namespace,
        capacity: usize,
    ) -> &mut SparseFeaturesNamespace {
        let item =
            self.namespaces
                .entry(namespace)
                .or_insert(SparseFeaturesNamespace::new_with_capacity(
                    namespace, capacity,
                ));
        item.set_active(true);
        item
    }

    pub fn append(&mut self, other: &SparseFeatures) {
        for (ns, feats) in &other.namespaces {
            if feats.active {
                let container = self.get_or_create_namespace_with_capacity(*ns, feats.size());
                container.add_features(&feats.feature_indices, &feats.feature_values);
            }
        }
    }

    // This function assumes no other objects were "appended" because it depends on truncating values off.
    // It also assumes that the other object has not changed since it was appended.
    pub fn remove(&mut self, other: &SparseFeatures) {
        for (ns, feats) in &other.namespaces {
            if feats.active {
                let container = self.get_or_create_namespace(*ns);
                let size = container.size();
                container.feature_indices.truncate(size - feats.size());
                container.feature_values.truncate(size - feats.size());

                // If the container is now empty, deactivate it.
                if container.size() == 0 {
                    container.set_active(false);
                }
            }
        }
    }

    pub fn empty(&self) -> bool {
        self.namespaces.is_empty() || self.namespaces.values().all(|ns| !ns.is_active())
    }
}

impl PoolReturnable<SparseFeatures> for SparseFeatures {
    fn clear_and_return_object(mut self, pool: &crate::object_pool::Pool<SparseFeatures>) {
        self.clear();
        pool.return_object(self);
    }
}
