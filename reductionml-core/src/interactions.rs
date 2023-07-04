use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{hash::hash_bytes, sparse_namespaced_features::Namespace};

#[derive(Serialize, Deserialize, Clone, JsonSchema, Debug)]
pub enum NamespaceDef {
    Name(String),
    Default,
}

pub type Interaction = Vec<NamespaceDef>;
pub type HashedInteraction = Vec<Namespace>;

pub fn hash_interaction(interaction: &Interaction, hash_seed: u32) -> HashedInteraction {
    if interaction.is_empty() || interaction.len() > 3 {
        panic!("Interaction must be between 1 and 3 namespaces")
    }
    interaction
        .iter()
        .map(|ns| match ns {
            NamespaceDef::Name(name) => {
                let namespace_hash = hash_bytes(name.as_bytes(), hash_seed);
                Namespace::Named(namespace_hash.into())
            }
            NamespaceDef::Default => Namespace::Default,
        })
        .collect()
}

type InteractionPair = (Namespace, Namespace);
type InteractionTriple = (Namespace, Namespace, Namespace);

pub fn compile_interactions(
    interactions: &[Interaction],
    hash_seed: u32,
) -> (Vec<InteractionPair>, Vec<InteractionTriple>) {
    let pairs: Vec<(Namespace, Namespace)> = interactions
        .iter()
        .filter(|interaction| interaction.len() == 2)
        .map(|interaction| {
            let hashed_interaction = hash_interaction(interaction, hash_seed);
            (hashed_interaction[0], hashed_interaction[1])
        })
        .collect();

    let triples: Vec<(Namespace, Namespace, Namespace)> = interactions
        .iter()
        .filter(|interaction| interaction.len() == 3)
        .map(|interaction| {
            let hashed_interaction = hash_interaction(interaction, hash_seed);
            (
                hashed_interaction[0],
                hashed_interaction[1],
                hashed_interaction[2],
            )
        })
        .collect();

    if pairs.len() + triples.len() != interactions.len() {
        panic!("Invalid interaction. Only pairs and triples are supported")
    }

    (pairs, triples)
}
