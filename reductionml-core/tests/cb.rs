use std::sync::Arc;

use reductionml_core::{
    interactions::NamespaceDef,
    object_pool::Pool,
    parsers::{JsonParserFactory, TextModeParser, TextModeParserFactory},
    sparse_namespaced_features::{Namespace, SparseFeatures},
    utils::AsInner, reductions::{CBExploreAdfGreedyConfig, CBExploreAdfGreedyReductionFactory}, global_config::GlobalConfig, reduction_factory::ReductionFactory, ActionProbsPrediction, reduction::DepthInfo, Label, CBLabel, CBAdfFeatures, Features, FeaturesType, LabelType,
};
use serde_json::json;


#[test]
fn test_greedy_predict() {
    let cb_adf_greedy_config = CBExploreAdfGreedyConfig::default();
    let global_config = GlobalConfig::new(8, 0, false, &Vec::new());
    let factory = CBExploreAdfGreedyReductionFactory::default();
    let mut cb_explore_adf_greedy = factory
        .create(&cb_adf_greedy_config, &global_config, 1.into())
        .unwrap();

    let shared_features = {
        let mut features = SparseFeatures::new();
        let ns = features.get_or_create_namespace(Namespace::Default);
        ns.add_feature(0.into(), 1.0);
        features
    };

    let actions = vec![
        {
            let mut features = SparseFeatures::new();
            let ns = features.get_or_create_namespace(Namespace::Default);
            ns.add_feature(0.into(), 1.0);
            features
        },
        {
            let mut features = SparseFeatures::new();
            let ns = features.get_or_create_namespace(Namespace::Default);
            ns.add_feature(0.into(), 1.0);
            features
        },
    ];

    let features = CBAdfFeatures {
        shared: Some(shared_features),
        actions,
    };

    let label = CBLabel {
        action: 0,
        cost: 0.0,
        probability: 1.0,
    };

    let mut features = Features::SparseCBAdf(features);
    let label = Label::CB(label);
    let mut depth_info = DepthInfo::new();
    let prediction = cb_explore_adf_greedy.predict_then_learn(
        &mut features,
        &label,
        &mut depth_info,
        0.into(),
    );
    let pred: &ActionProbsPrediction = prediction.as_inner().unwrap();
    assert!(pred.0.len() == 2);
}

#[test]
fn test_greedy_predict_json() {
    let cb_adf_greedy_config = CBExploreAdfGreedyConfig::default();
    let global_config = GlobalConfig::new(8, 0, false, &Vec::new());
    let factory = CBExploreAdfGreedyReductionFactory::default();
    let mut cb_explore_adf_greedy = factory
        .create(&cb_adf_greedy_config, &global_config, 1.into())
        .unwrap();

    let pool = Arc::new(Pool::new());
    let json_parser_factory = JsonParserFactory::default();
    let json_parser = json_parser_factory.create(
        FeaturesType::SparseCBAdf,
        LabelType::CB,
        0,
        global_config.num_bits(),
        pool.clone(),
    );

    let input = json!({
        "label": {
            "action": 0,
            "cost": 1.0,
            "probability": 1.0
        },
        "shared": {
            "shared_ns": {
                "test": 1.0,
            }
        },
        "actions": [
            {
                "action_ns": {
                    "test1": 1.0
                }
            },
            {
                "action_ns": {
                    "test2": 1.0
                }
            }
        ]
    });
    let (mut features, label) = json_parser.parse_chunk(&input.to_string()).unwrap();

    let mut depth_info = DepthInfo::new();
    let prediction = cb_explore_adf_greedy.predict_then_learn(
        &mut features,
        &label.unwrap(),
        &mut depth_info,
        0.into(),
    );
    let pred: &ActionProbsPrediction = prediction.as_inner().unwrap();
    assert!(pred.0.len() == 2);
}