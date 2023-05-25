use crate::{object_pool::PoolReturnable, FeatureIndex, NamespaceHash};

pub struct NamespacesIterator<'a> {
    namespaces: std::collections::hash_map::Iter<'a, Namespace, SparseFeaturesNamespace>,
}

pub struct NamespaceIterator<'a> {
    indices: std::slice::Iter<'a, FeatureIndex>,
    values: std::slice::Iter<'a, f32>,
}

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
}

#[derive(PartialEq, Clone)]
pub struct SparseFeaturesNamespace {
    namespace: Namespace,
    feature_indices: Vec<FeatureIndex>,
    feature_values: Vec<f32>,
    active: bool,
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace {
    Named(NamespaceHash),
    Default,
}

#[derive(PartialEq, Clone)]
pub struct SparseFeatures {
    namespaces: std::collections::HashMap<Namespace, SparseFeaturesNamespace>,
}

impl Default for SparseFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseFeatures {
    pub fn namespaces(&self) -> NamespacesIterator {
        NamespacesIterator {
            namespaces: self.namespaces.iter(),
        }
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
}

impl PoolReturnable<SparseFeatures> for SparseFeatures {
    fn return_object(self, pool: &crate::object_pool::Pool<SparseFeatures>) {
        pool.return_object(self);
    }
}
