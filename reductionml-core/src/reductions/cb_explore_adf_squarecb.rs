use crate::error::Result;
use crate::global_config::GlobalConfig;

use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, PascalCaseString, ReductionConfig, ReductionFactory,
};

use crate::utils::AsInner;
use crate::{impl_default_factory_functions, types::*, ModelIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_json::json;

use crate::explore::enforce_min_prob;

use super::{CBAdfConfig, CBType};

#[derive(Clone, Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct CBExploreAdfSquareCBConfig {
    #[serde(default = "default_uniform_epsilon")]
    uniform_epsilon: f32,

    #[serde(default = "default_gamma_scale")]
    gamma_scale: f32,

    #[serde(default = "default_gamma_exponent")]
    gamma_exponent: f32,

    #[serde(default = "default_cb_adf")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    cb_adf: JsonReductionConfig,
}

fn default_uniform_epsilon() -> f32 {
    0.0
}

fn default_gamma_scale() -> f32 {
    10.0
}

fn default_gamma_exponent() -> f32 {
    0.5
}

fn default_cb_adf() -> JsonReductionConfig {
    JsonReductionConfig::new("CbAdf".try_into().unwrap(), json!(CBAdfConfig::default()))
}

impl ReductionConfig for CBExploreAdfSquareCBConfig {
    fn typename(&self) -> PascalCaseString {
        "CbExploreAdfSquareCb".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
struct CBExploreAdfSquareCBReduction {
    config: CBExploreAdfSquareCBConfig,
    counter: usize,
    cb_adf: ReductionWrapper,
}

#[derive(Default)]
pub struct CBExploreAdfSquareCBReductionFactory;

impl ReductionFactory for CBExploreAdfSquareCBReductionFactory {
    impl_default_factory_functions!("CbExploreAdfSquareCb", CBExploreAdfSquareCBConfig);

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<CBExploreAdfSquareCBConfig>()
            .unwrap();
        let cb_adf_config = crate::reduction_factory::parse_config(&config.cb_adf)?;
        // Try checking for MTR
        if let Some(cfg) = cb_adf_config.as_any().downcast_ref::<CBAdfConfig>() {
            if cfg.cb_type() != CBType::Mtr {
                return Err(crate::error::Error::InvalidArgument(
                    "CBExploreAdfSquareCB only supports CB implementatons using cb type MTR"
                        .to_string(),
                ));
            }
        }

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
            Box::new(CBExploreAdfSquareCBReduction {
                config: config.clone(),
                counter: 0,
                cb_adf,
            }),
            types,
            num_models_above,
        ))
    }
}

#[typetag::serde]
impl ReductionImpl for CBExploreAdfSquareCBReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let gamma =
            self.config.gamma_scale * (self.counter as f32).powf(self.config.gamma_exponent);

        let pred = self.cb_adf.predict(features, depth_info, 0.into());
        let mut scores: ActionScoresPrediction = pred.try_into().unwrap();
        let best_idx = scores
            .0
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let best_score = scores.0[best_idx].1;
        let best_action = scores.0[best_idx].0;

        let mut total_weight = 0.0;

        let num_actions = scores.0.len();
        for (action, score) in scores.0.iter_mut() {
            if *action == best_action {
                continue;
            }
            *score = 1.0 / ((num_actions as f32) + gamma * (*score - best_score));
            total_weight += *score;
        }

        scores.0[best_idx].1 = 1.0 - total_weight;

        enforce_min_prob(self.config.uniform_epsilon, false, &mut scores.0).unwrap();

        Prediction::ActionProbs(ActionProbsPrediction(scores.0))
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let cb_label: &CBLabel = label.as_inner().unwrap();
        let mut cb_label_clone: CBLabel = *cb_label;
        cb_label_clone.probability = 1.0;
        let new_label = Label::CB(cb_label_clone);
        self.cb_adf
            .learn(features, &new_label, depth_info, 0.into());
        self.counter += 1;
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.cb_adf]
    }
}
