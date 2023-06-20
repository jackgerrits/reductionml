use crate::error::{Result};
use crate::global_config::GlobalConfig;
use crate::object_pool::{Pool, PoolReturnable};
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, ReductionConfig, ReductionFactory,
};
use crate::utils::GetInner;

use crate::reductions::CoinRegressorConfig;
use crate::sparse_namespaced_features::SparseFeatures;
use crate::{impl_default_factory_functions, types::*, ModelIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Copy, JsonSchema)]
enum CBType {
    #[serde(rename = "ips")]
    Ips,
    #[serde(rename = "mtr")]
    Mtr,
}

#[derive(Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
pub struct CBAdfConfig {
    #[serde(default = "default_cb_type")]
    cb_type: CBType,
    #[serde(default = "default_regressor")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    regressor: JsonReductionConfig,
}

fn default_cb_type() -> CBType {
    CBType::Ips
}

impl ReductionConfig for CBAdfConfig {
    fn typename(&self) -> String {
        "cb_adf".to_owned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn default_regressor() -> JsonReductionConfig {
    JsonReductionConfig::new("coin".to_owned(), json!(CoinRegressorConfig::default()))
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
    #[serde(skip)]
    object_pool: Pool<SparseFeatures>,
    // TODO: have MTR state per interleaved model.
    mtr_state: MtrState,
}

#[derive(Default)]
pub struct CBAdfReductionFactory;

impl ReductionFactory for CBAdfReductionFactory {
    impl_default_factory_functions!("cb_adf", CBAdfConfig);

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
                object_pool: Default::default(),
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
    fn predict(&self, features: &Features, depth_info: &mut DepthInfo) -> Prediction {
        let cb_adf_features: &CBAdfFeatures = features.get_inner_ref().unwrap();

        let mut feats_to_reuse = self.object_pool.get_object();

        if let Some(shared_feats) = &cb_adf_features.shared {
            feats_to_reuse.append(shared_feats);
        }

        let mut action_scores: ActionScoresPrediction = Default::default();

        let mut counter = 0;
        for action in &cb_adf_features.actions {
            feats_to_reuse.append(action);
            let wrapped_feats = feats_to_reuse.into();
            let pred = self.regressor.predict(&wrapped_feats, depth_info, 0.into());
            let scalar_pred: &ScalarPrediction = pred.get_inner_ref().unwrap();
            action_scores.0.push((counter, scalar_pred.raw_prediction));
            feats_to_reuse = SparseFeatures::try_from(wrapped_feats).unwrap();
            feats_to_reuse.remove(action);
            counter += 1;
        }

        feats_to_reuse.clear_and_return_object(&self.object_pool);

        action_scores.into()
    }

    fn predict_then_learn(
        &mut self,
        _features: &Features,
        _label: &Label,
        _depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        todo!()
    }

    fn learn(
        &mut self,
        features: &Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let cb_adf_features: &CBAdfFeatures = features.get_inner_ref().unwrap();
        let cb_label: &CBLabel = label.get_inner_ref().unwrap();

        match self.cb_type {
            CBType::Ips => {
                let mut feats_to_reuse = self.object_pool.get_object();

                if let Some(shared_feats) = &cb_adf_features.shared {
                    feats_to_reuse.append(shared_feats);
                }

                let mut counter = 0;
                for action in &cb_adf_features.actions {
                    feats_to_reuse.append(action);
                    let wrapped_feats = feats_to_reuse.into();

                    self.regressor.learn(
                        &wrapped_feats,
                        &(generate_ips_simple_label(cb_label, counter).into()),
                        depth_info,
                        0.into(),
                    );
                    feats_to_reuse = SparseFeatures::try_from(wrapped_feats).unwrap();
                    feats_to_reuse.remove(action);
                    counter += 1;
                }

                feats_to_reuse.clear_and_return_object(&self.object_pool);
            }
            CBType::Mtr => {
                self.mtr_state.action_sum += cb_adf_features.actions.len();
                self.mtr_state.event_sum += 1;

                let cost = cb_label.cost;
                // TODO clip_p
                let prob = cb_label.probability;
                let weight = 1.0 / prob
                    * (self.mtr_state.event_sum as f32 / self.mtr_state.action_sum as f32);

                let simple_label = SimpleLabel(cost, weight);
                match &cb_adf_features.shared {
                    Some(shared_feats) => {
                        let mut feats_obj = self.object_pool.get_object();
                        feats_obj.append(shared_feats);
                        feats_obj.append(cb_adf_features.actions.get(cb_label.action).unwrap());
                        self.regressor.learn(
                            &Features::SparseSimpleRef(&feats_obj),
                            &simple_label.into(),
                            depth_info,
                            0.into(),
                        );
                        feats_obj.clear_and_return_object(&self.object_pool);
                    }
                    None => {
                        todo!()
                    }
                }
                // let mut feats_to_reuse =
                //     self.object_pool.pull();
            }
        }
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.regressor]
    }
}
