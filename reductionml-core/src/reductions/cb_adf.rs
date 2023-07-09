use crate::error::Result;
use crate::global_config::GlobalConfig;
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, PascalCaseString, ReductionConfig, ReductionFactory,
};
use crate::utils::AsInner;

use crate::reductions::CoinRegressorConfig;
use crate::{impl_default_factory_functions, types::*, ModelIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Copy, JsonSchema, PartialEq)]
pub enum CBType {
    #[serde(rename = "ips")]
    Ips,
    #[serde(rename = "mtr")]
    Mtr,
}

#[derive(Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct CBAdfConfig {
    #[serde(default = "default_cb_type")]
    cb_type: CBType,
    #[serde(default = "default_regressor")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    regressor: JsonReductionConfig,
}

impl CBAdfConfig {
    pub fn cb_type(&self) -> CBType {
        self.cb_type
    }
}

fn default_cb_type() -> CBType {
    CBType::Mtr
}

impl ReductionConfig for CBAdfConfig {
    fn typename(&self) -> PascalCaseString {
        "CbAdf".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn default_regressor() -> JsonReductionConfig {
    JsonReductionConfig::new(
        "Coin".try_into().unwrap(),
        json!(CoinRegressorConfig::default()),
    )
}

#[derive(Serialize, Deserialize, Default)]
struct MtrState {
    action_sum: usize,
    event_sum: usize,
}

#[derive(Serialize, Deserialize)]
struct CBAdfReduction {
    cb_type: CBType,
    regressor: ReductionWrapper,
    // TODO: have MTR state per interleaved model.
    mtr_state: MtrState,
}

#[derive(Default)]
pub struct CBAdfReductionFactory;

impl ReductionFactory for CBAdfReductionFactory {
    impl_default_factory_functions!("CbAdf", CBAdfConfig);

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config.as_any().downcast_ref::<CBAdfConfig>().unwrap();
        let regressor_config = crate::reduction_factory::parse_config(&config.regressor)?;
        let regressor: ReductionWrapper =
            create_reduction(regressor_config.as_ref(), global_config, num_models_above)?;

        let types = ReductionTypeDescriptionBuilder::new(
            LabelType::CB,
            FeaturesType::SparseCBAdf,
            PredictionType::ActionScores,
        )
        .with_input_prediction_type(PredictionType::Scalar)
        .with_output_features_type(FeaturesType::SparseSimple)
        .with_output_label_type(LabelType::Simple)
        .build();

        if let Some(reason) = types.check_and_get_reason(regressor.types()) {
            return Err(crate::error::Error::InvalidArgument(format!(
                "Invalid reduction configuration: {}",
                reason
            )));
        }

        Ok(ReductionWrapper::new(
            self.typename(),
            Box::new(CBAdfReduction {
                cb_type: config.cb_type,
                regressor,
                mtr_state: Default::default(),
            }),
            types,
            num_models_above,
        ))
    }
}

// TODO: clip_p
fn generate_ips_simple_label(label: &CBLabel, current_action_index: usize) -> SimpleLabel {
    if current_action_index == label.action {
        debug_assert!(label.probability > 0.0);
        (label.cost / label.probability).into()
    } else {
        0.0.into()
    }
}

#[typetag::serde]
impl ReductionImpl for CBAdfReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let cb_adf_features: &mut CBAdfFeatures = features.as_inner_mut().unwrap();

        let mut action_scores = ActionScoresPrediction::default();
        for (counter, action) in cb_adf_features.actions.iter_mut().enumerate() {
            if let Some(shared_feats) = &cb_adf_features.shared {
                action.append(shared_feats);
            }

            let pred = self
                .regressor
                .predict(&mut action.into(), depth_info, 0.into());
            let scalar_pred: &ScalarPrediction = pred.as_inner().unwrap();
            action_scores.0.push((counter, scalar_pred.raw_prediction));
            if let Some(shared_feats) = &cb_adf_features.shared {
                action.remove(shared_feats);
            }
        }

        action_scores.into()
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let cb_adf_features: &mut CBAdfFeatures = features.as_inner_mut().unwrap();
        let cb_label: &CBLabel = label.as_inner().unwrap();

        match self.cb_type {
            CBType::Ips => {
                for (counter, action) in cb_adf_features.actions.iter_mut().enumerate() {
                    if let Some(shared_feats) = &cb_adf_features.shared {
                        action.append(shared_feats);
                    }

                    self.regressor.learn(
                        &mut action.into(),
                        &(generate_ips_simple_label(cb_label, counter).into()),
                        depth_info,
                        0.into(),
                    );
                    if let Some(shared_feats) = &cb_adf_features.shared {
                        action.remove(shared_feats);
                    }
                }
            }
            CBType::Mtr => {
                self.mtr_state.action_sum += cb_adf_features.actions.len();
                self.mtr_state.event_sum += 1;

                let cost = cb_label.cost;
                // TODO clip_p
                let prob = cb_label.probability;
                let weight = 1.0 / prob
                    * (self.mtr_state.event_sum as f32 / self.mtr_state.action_sum as f32);

                let simple_label = SimpleLabel::new(cost, weight);
                match cb_adf_features.shared.as_mut() {
                    Some(shared_feats) => {
                        let action = cb_adf_features.actions.get(cb_label.action).unwrap();
                        shared_feats.append(action);
                        self.regressor.learn(
                            &mut Features::SparseSimpleRef(shared_feats),
                            &simple_label.into(),
                            depth_info,
                            0.into(),
                        );
                        shared_feats.remove(action);
                    }
                    None => {
                        todo!()
                    }
                }
            }
        }
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.regressor]
    }
}
