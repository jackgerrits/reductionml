use crate::error::Result;
use crate::global_config::GlobalConfig;

use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, PascalCaseString, ReductionConfig, ReductionFactory,
};

use crate::explore::enforce_min_prob;
use crate::{impl_default_factory_functions, types::*, ModelIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_json::json;

use super::CBAdfConfig;

#[derive(Clone, Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct CBExploreAdfSoftmaxConfig {
    #[serde(default = "default_uniform_epsilon")]
    uniform_epsilon: f32,

    #[serde(default = "default_lambda")]
    lambda: f32,

    #[serde(default = "default_cb_adf")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    cb_adf: JsonReductionConfig,
}

fn default_uniform_epsilon() -> f32 {
    0.0
}

fn default_lambda() -> f32 {
    1.0
}

fn default_cb_adf() -> JsonReductionConfig {
    JsonReductionConfig::new("CbAdf".try_into().unwrap(), json!(CBAdfConfig::default()))
}

impl ReductionConfig for CBExploreAdfSoftmaxConfig {
    fn typename(&self) -> PascalCaseString {
        "CbExploreAdfSoftmax".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
struct CBExploreAdfSoftmaxReduction {
    config: CBExploreAdfSoftmaxConfig,
    counter: usize,
    cb_adf: ReductionWrapper,
}

#[derive(Default)]
pub struct CBExploreAdfSoftmaxReductionFactory;

impl ReductionFactory for CBExploreAdfSoftmaxReductionFactory {
    impl_default_factory_functions!("CbExploreAdfSoftmax", CBExploreAdfSoftmaxConfig);

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<CBExploreAdfSoftmaxConfig>()
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
            Box::new(CBExploreAdfSoftmaxReduction {
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
impl ReductionImpl for CBExploreAdfSoftmaxReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
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

        let mut total_weight = 0.0;

        for (_, score) in scores.0.iter_mut() {
            *score = (-self.config.lambda * (*score - best_score)).exp();
            total_weight += *score;
        }
        for (_, score) in scores.0.iter_mut() {
            *score /= total_weight;
        }

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
        self.cb_adf.learn(features, label, depth_info, 0.into());
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.cb_adf]
    }
}
