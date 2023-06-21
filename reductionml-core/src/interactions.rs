use std::io::Cursor;

use murmur3::murmur3_32;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::sparse_namespaced_features::Namespace;

#[derive(Serialize, Deserialize, Clone, JsonSchema, Debug)]
pub enum NamespaceDef {
    Named(String),
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
            NamespaceDef::Named(name) => {
                let namespace_hash = murmur3_32(&mut Cursor::new(name), hash_seed).unwrap();
                Namespace::Named(namespace_hash.into())
            }
            NamespaceDef::Default => Namespace::Default,
        })
        .collect()
}

type InteractionPair = (Namespace, Namespace);
type InteractionTriple = (Namespace, Namespace, Namespace);

pub fn compile_interactions(
    interactions: &Vec<Interaction>,
    hash_seed: u32,
) -> (Option<Vec<InteractionPair>>, Option<Vec<InteractionTriple>>) {
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

    let pairs = if !pairs.is_empty() { Some(pairs) } else { None };
    let triples = if !triples.is_empty() {
        Some(triples)
    } else {
        None
    };

    (pairs, triples)
}
