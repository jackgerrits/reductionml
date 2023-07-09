use std::sync::Arc;

use reductionml_core::{
    global_config::GlobalConfig,
    interactions::NamespaceDef,
    object_pool::Pool,
    parsers::{JsonParserFactory, TextModeParser, TextModeParserFactory},
    reduction::DepthInfo,
    reduction::ReductionWrapper,
    reduction_factory::ReductionFactory,
    reductions::{
        CBExploreAdfGreedyConfig, CBExploreAdfGreedyReductionFactory, CBExploreAdfSquareCBConfig,
        CBExploreAdfSquareCBReductionFactory,
    },
    sparse_namespaced_features::{Namespace, SparseFeatures},
    utils::AsInner,
    ActionProbsPrediction, CBAdfFeatures, CBLabel, Features, FeaturesType, Label, LabelType,
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
    let prediction =
        cb_explore_adf_greedy.predict_then_learn(&mut features, &label, &mut depth_info, 0.into());
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

fn ex_learn(
    context: &str,
    action0: &str,
    action1: &str,
    chosen: i32,
    p: f32,
    reward: f32,
) -> String {
    json!({
        "label": {
            "action": chosen,
            "cost": -reward,
            "probability": p
        },
        "shared": {
            "user": {
                "f": context,
            }
        },
        "actions": [
            {
                "content": {
                    "f": action0
                }
            },
            {
                "content": {
                    "f": action1
                }
            }
        ]
    })
    .to_string()
}

fn ex_pred(context: &str, action0: &str, action1: &str) -> String {
    json!({
        "shared": {
            "user": {
                "f": context,
            }
        },
        "actions": [
            {
                "content": {
                    "f": action0
                }
            },
            {
                "content": {
                    "f": action1
                }
            }
        ]
    })
    .to_string()
}

fn test_learning_e2e(
    context: fn(i32) -> String,
    action0: &str,
    action1: &str,
    chosen: fn(&str, i32) -> (i32, f32),
    r: fn(&str, &str, i32) -> f32,
    n: i32,
    learners: &mut [ReductionWrapper],
    global_config: &GlobalConfig,
    test_set: &[(String, usize)],
) {
    for mut learner in learners.iter_mut() {
        let pool = Arc::new(Pool::new());
        let json_parser_factory = JsonParserFactory::default();
        let json_parser = json_parser_factory.create(
            FeaturesType::SparseCBAdf,
            LabelType::CB,
            0,
            global_config.num_bits(),
            pool.clone(),
        );
        let actions = [action0, action1];
        for i in 0..n {
            let ctx = context(i);
            let (chosen_action, p) = chosen(&ctx, i);
            let reward = r(&ctx, &actions[usize::try_from(chosen_action).unwrap()], i);
            let (mut features, label) = json_parser
                .parse_chunk(&ex_learn(&ctx, action0, action1, chosen_action, p, reward))
                .unwrap();

            let mut depth_info = DepthInfo::new();
            learner.learn(&mut features, &label.unwrap(), &mut depth_info, 0.into());
        }
        for (ctx, expected) in test_set {
            let (mut features, _) = json_parser
                .parse_chunk(&ex_pred(&ctx, action0, action1))
                .unwrap();
            let prediction = learner.predict(&mut features, &mut DepthInfo::new(), 0.into());
            let pred: &ActionProbsPrediction = prediction.as_inner().unwrap();
            for (action, prob) in pred.0.iter() {
                if action.eq(expected) {
                    assert!(
                        prob > &0.5,
                        "learner: {}, ctx: {}, expected exploit: {}, actual prob: {}",
                        learner.typename(),
                        ctx,
                        expected,
                        prob
                    );
                }
            }
        }
    }
}

#[test]
fn test_cb_stationary_deterministic_actions_single_context() {
    fn context(i: i32) -> String {
        "Tom".to_owned()
    }
    fn chosen(context: &str, i: i32) -> (i32, f32) {
        (i % 2, 0.5)
    }
    fn r(context: &str, action: &str, i: i32) -> f32 {
        if action == "Politics" {
            1.0
        } else {
            0.0
        }
    }

    let global_config = GlobalConfig::new(5, 0, true, &Vec::new());
    let mut learners = [
        CBExploreAdfGreedyReductionFactory::default()
            .create(
                &CBExploreAdfGreedyConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
        CBExploreAdfSquareCBReductionFactory::default()
            .create(
                &CBExploreAdfSquareCBConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
    ];

    test_learning_e2e(
        context,
        "Politics",
        "Sports",
        chosen,
        r,
        1000,
        &mut learners,
        &global_config,
        &[("Tom".to_owned(), 0)],
    );
}

#[test]
fn test_cb_stationary_deterministic_actions_with_personalization() {
    fn context(i: i32) -> String {
        if i % 4 < 2 {
            "Tom".to_owned()
        } else {
            "Anna".to_owned()
        }
    }
    fn chosen(context: &str, i: i32) -> (i32, f32) {
        (i % 2, 0.5)
    }
    fn r(context: &str, action: &str, i: i32) -> f32 {
        if context == "Tom" && action == "Politics" {
            1.0
        } else if context == "Tom" && action == "Sports" {
            0.0
        } else if context == "Anna" && action == "Politics" {
            0.0
        } else {
            1.0
        }
    }

    let global_config = GlobalConfig::new(
        5,
        0,
        true,
        &vec![vec![
            NamespaceDef::Name("user".to_owned()),
            NamespaceDef::Name("content".to_owned()),
        ]],
    );
    let mut learners = [
        CBExploreAdfGreedyReductionFactory::default()
            .create(
                &CBExploreAdfGreedyConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
        CBExploreAdfSquareCBReductionFactory::default()
            .create(
                &CBExploreAdfSquareCBConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
    ];

    test_learning_e2e(
        context,
        "Politics",
        "Sports",
        chosen,
        r,
        1000,
        &mut learners,
        &global_config,
        &[("Tom".to_owned(), 0), ("Anna".to_owned(), 1)],
    );
}

#[test]
fn test_cb_nonstationary_deterministic_actions_with_personalization() {
    fn context(i: i32) -> String {
        if i % 4 < 2 {
            "Tom".to_owned()
        } else {
            "Anna".to_owned()
        }
    }
    fn chosen(context: &str, i: i32) -> (i32, f32) {
        (i % 2, 0.5)
    }
    fn r(context: &str, action: &str, i: i32) -> f32 {
        if i < 1000 {
            if context == "Tom" && action == "Politics" {
                1.0
            } else if context == "Tom" && action == "Sports" {
                0.0
            } else if context == "Anna" && action == "Politics" {
                0.0
            } else {
                1.0
            }
        } else {
            if context == "Tom" && action == "Politics" {
                0.0
            } else if context == "Tom" && action == "Sports" {
                1.0
            } else if context == "Anna" && action == "Politics" {
                1.0
            } else {
                0.0
            }
        }
    }

    let global_config = GlobalConfig::new(
        5,
        0,
        true,
        &vec![vec![
            NamespaceDef::Name("user".to_owned()),
            NamespaceDef::Name("content".to_owned()),
        ]],
    );
    let mut learners = [
        CBExploreAdfGreedyReductionFactory::default()
            .create(
                &CBExploreAdfGreedyConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
        CBExploreAdfSquareCBReductionFactory::default()
            .create(
                &CBExploreAdfSquareCBConfig::default(),
                &global_config,
                1.into(),
            )
            .unwrap(),
    ];

    test_learning_e2e(
        context,
        "Politics",
        "Sports",
        chosen,
        r,
        2000,
        &mut learners,
        &global_config,
        &[("Tom".to_owned(), 1), ("Anna".to_owned(), 0)],
    );
}
