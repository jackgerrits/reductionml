use crate::error::Result;
use crate::global_config::GlobalConfig;

use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, PascalCaseString, ReductionConfig, ReductionFactory,
};

use crate::{impl_default_factory_functions, types::*, ModelIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_json::json;

use super::CBAdfConfig;

#[derive(Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct CBExploreAdfGreedyConfig {
    #[serde(default = "default_epsilon")]
    epsilon: f32,

    #[serde(default = "default_cb_adf")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    cb_adf: JsonReductionConfig,
}

fn default_epsilon() -> f32 {
    0.05
}

fn default_cb_adf() -> JsonReductionConfig {
    JsonReductionConfig::new("CbAdf".try_into().unwrap(), json!(CBAdfConfig::default()))
}

impl ReductionConfig for CBExploreAdfGreedyConfig {
    fn typename(&self) -> PascalCaseString {
        "CbExploreAdfGreedy".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
struct CBExploreAdfGreedyReduction {
    epsilon: f32,
    cb_adf: ReductionWrapper,
}

#[derive(Default)]
pub struct CBExploreAdfGreedyReductionFactory;

impl ReductionFactory for CBExploreAdfGreedyReductionFactory {
    impl_default_factory_functions!("CbExploreAdfGreedy", CBExploreAdfGreedyConfig);

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<CBExploreAdfGreedyConfig>()
            .unwrap();
        let cb_adf_config = crate::reduction_factory::parse_config(&config.cb_adf)?;
        let cb_adf: ReductionWrapper =
            create_reduction(cb_adf_config.as_ref(), global_config, num_models_above)?;

        let types = ReductionTypeDescriptionBuilder::new(
            LabelType::CB,
            FeaturesType::SparseCBAdf,
            PredictionType::ActionProbs,
        )
        .with_input_prediction_type(PredictionType::ActionScores)
        .with_output_features_type(FeaturesType::SparseCBAdf)
        .with_output_label_type(LabelType::CB)
        .build();

        if let Some(reason) = types.check_and_get_reason(cb_adf.types()) {
            return Err(crate::error::Error::InvalidArgument(format!(
                "Invalid reduction configuration: {}",
                reason
            )));
        }

        Ok(ReductionWrapper::new(
            self.typename(),
            Box::new(CBExploreAdfGreedyReduction {
                epsilon: config.epsilon,
                cb_adf,
            }),
            types,
            num_models_above,
        ))
    }
}

impl CBExploreAdfGreedyReduction {
    // VW::action_scores& preds = examples[0]->pred.a_s;
    // uint32_t num_actions = static_cast<uint32_t>(preds.size());

    // auto& ep_fts = examples[0]->ex_reduction_features.template get<VW::cb_explore_adf::greedy::reduction_features>();
    // float actual_ep = (ep_fts.valid_epsilon_supplied()) ? ep_fts.epsilon : _epsilon;

    // size_t tied_actions = fill_tied(preds);

    // const float prob = actual_ep / num_actions;
    // for (size_t i = 0; i < num_actions; i++) { preds[i].score = prob; }
    // if (!_first_only)
    // {
    //   for (size_t i = 0; i < tied_actions; ++i) { preds[i].score += (1.f - actual_ep) / tied_actions; }
    // }
    // else { preds[0].score += 1.f - actual_ep; }
    fn action_scores_to_probs(
        &self,
        mut action_scores: ActionScoresPrediction,
    ) -> ActionProbsPrediction {
        let best_action = action_scores
            .0
            .iter()
            .enumerate()
            .min_by(|(_, (_, a)), (_, (_, b))| {
                if a > b {
                    std::cmp::Ordering::Greater
                } else if a < b {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .unwrap()
            .0;

        let equal_prob = self.epsilon / action_scores.0.len() as f32;
        action_scores
            .0
            .iter_mut()
            .for_each(|(_, p)| *p = equal_prob);
        action_scores.0[best_action].1 += 1.0 - self.epsilon;

        ActionProbsPrediction(action_scores.0)
    }
}

#[typetag::serde]
impl ReductionImpl for CBExploreAdfGreedyReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let pred = self.cb_adf.predict(features, depth_info, 0.into());
        let scores: ActionScoresPrediction = pred.try_into().unwrap();
        Prediction::ActionProbs(self.action_scores_to_probs(scores))
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        self.cb_adf.learn(features, label, depth_info, 0.into());
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.cb_adf]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;

    use crate::{
        interactions::NamespaceDef,
        object_pool::Pool,
        parsers::{JsonParserFactory, TextModeParser, TextModeParserFactory},
        sparse_namespaced_features::{Namespace, SparseFeatures},
        utils::AsInner,
    };

    use super::*;

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
}
