// class ftrl_update_data
// {
// public:
//   float update = 0.f;
//   float ftrl_alpha = 0.f;
//   float ftrl_beta = 0.f;
//   float l1_lambda = 0.f;
//   float l2_lambda = 0.f;
//   float predict = 0.f;
//   float normalized_squared_norm_x = 0.f;
//   float average_squared_norm_x = 0.f;
// };

// class ftrl
// {
// public:
//   VW::workspace* all = nullptr;  // features, finalize, l1, l2,
//   float ftrl_alpha = 0.f;
//   float ftrl_beta = 0.f;
//   ftrl_update_data data;
//   uint32_t ftrl_size = 0;
//   std::vector<VW::reductions::details::gd_per_model_state> gd_per_model_states;
// };

use std::ops::Deref;

use crate::dense_weights::DenseWeights;
use crate::error::Result;
use crate::global_config::GlobalConfig;
use crate::loss_function::{LossFunction, LossFunctionType};
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{ReductionConfig, ReductionFactory};
use crate::sparse_namespaced_features::SparseFeatures;
use crate::utils::bits_to_max_feature_index;
use crate::utils::GetInner;
use crate::weights::{foreach_feature, foreach_feature_with_state, foreach_feature_with_state_mut};
use crate::{types::*, ModelIndex, StateIndex};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;

#[derive(Deserialize, DefaultFromSerde, Serialize, Debug, Clone)]
pub struct CoinRegressorConfig {
    #[serde(default = "default_alpha")]
    alpha: f32,

    #[serde(default = "default_beta")]
    beta: f32,

    #[serde(default)]
    l1_lambda: f32,

    #[serde(default)]
    l2_lambda: f32,
}

const fn default_alpha() -> f32 {
    4.0
}

const fn default_beta() -> f32 {
    1.0
}

impl ReductionConfig for CoinRegressorConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn typename(&self) -> String {
        "coin".to_owned()
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CoinRegressorModelState {
    normalized_sum_norm_x: f32,
    total_weight: f32,
}

struct LossFunctionHolder {
    loss_function: Box<dyn LossFunction>,
}

impl Deref for LossFunctionHolder {
    type Target = dyn LossFunction;

    fn deref(&self) -> &Self::Target {
        self.loss_function.deref()
    }
}

impl Serialize for LossFunctionHolder {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.loss_function.get_type().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LossFunctionHolder {
    fn deserialize<D>(deserializer: D) -> std::result::Result<LossFunctionHolder, D::Error>
    where
        D: Deserializer<'de>,
    {
        LossFunctionType::deserialize(deserializer).map(|x| LossFunctionHolder {
            loss_function: x.create(),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct CoinRegressor {
    weights: DenseWeights,
    config: CoinRegressorConfig,
    model_states: Vec<CoinRegressorModelState>,
    average_squared_norm_x: f32,
    min_label: f32,
    max_label: f32,
    loss_function: LossFunctionHolder,
}

impl CoinRegressor {
    pub fn new(
        config: CoinRegressorConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<CoinRegressor> {
        Ok(CoinRegressor {
            weights: DenseWeights::new(
                bits_to_max_feature_index(global_config.num_bits()),
                num_models_above,
                StateIndex::from(6),
            )?,
            config,
            model_states: vec![
                CoinRegressorModelState {
                    normalized_sum_norm_x: 0.0,
                    total_weight: 0.0
                };
                *num_models_above as usize
            ],
            average_squared_norm_x: 0.0,
            min_label: 0.0,
            max_label: 0.0,
            loss_function: LossFunctionHolder {
                loss_function: LossFunctionType::Squared.create(),
            },
        })
    }
}

#[derive(Default)]
pub struct CoinRegressorFactory;

impl ReductionFactory for CoinRegressorFactory {
    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<CoinRegressorConfig>()
            .unwrap();

        Ok(ReductionWrapper::new(
            Box::new(CoinRegressor::new(
                config.clone(),
                global_config,
                num_models_above,
            )?),
            ReductionTypeDescriptionBuilder::new(
                LabelType::Simple,
                FeaturesType::SparseSimple,
                PredictionType::Scalar,
            )
            .build(),
            num_models_above,
        ))
    }

    fn typename(&self) -> String {
        "coin".to_owned()
    }
    fn parse_config(&self, value: &serde_json::Value) -> Result<Box<dyn ReductionConfig>> {
        let res: CoinRegressorConfig = serde_json::from_value(value.clone()).unwrap();
        Ok(Box::new(res))
    }
}

#[typetag::serde]
impl ReductionImpl for CoinRegressor {
    fn predict(&self, features: &Features, depth_info: &mut DepthInfo) -> Prediction {
        let sparse_feats: &SparseFeatures = features.get_inner_ref().unwrap();
        let mut prediction = 0.0;
        foreach_feature(
            depth_info.absolute_offset(),
            sparse_feats,
            &self.weights,
            |feat_val, weight_val| prediction += feat_val * weight_val,
        );

        let scalar_pred = ScalarPrediction {
            prediction: prediction.clamp(self.min_label, self.max_label),
            raw_prediction: prediction,
        };
        scalar_pred.into()
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
        _depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let sparse_feats: &SparseFeatures = features.get_inner_ref().unwrap();
        let simple_label: &SimpleLabel = label.get_inner_ref().unwrap();

        self.min_label = simple_label.0.min(self.min_label);
        self.max_label = simple_label.0.max(self.max_label);
        let _prediction = self.coin_betting_predict(sparse_feats, simple_label.1);
        self.coin_betting_update_after_predict(
            sparse_feats,
            _prediction,
            simple_label.0,
            simple_label.1,
        );
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![]
    }

    fn sensitivity(
        &self,
        _features: &Features,
        _label: f32,
        _prediction: f32,
        _weight: f32,
        _depth_info: DepthInfo,
    ) -> f32 {
        todo!()
    }

    fn typename(&self) -> String {
        "coin".to_owned()
    }
}

const W_XT: usize = 0; //  current parameter
const W_ZT: usize = 1; //  sum negative gradients
const W_G2: usize = 2; //  sum of absolute value of gradients
const W_MX: usize = 3; //  maximum absolute value
const W_WE: usize = 4; //  Wealth
const W_MG: usize = 5; //  Maximum Lipschitz constant

impl CoinRegressor {
    fn coin_betting_predict(&mut self, features: &SparseFeatures, weight: f32) -> f32 {
        let mut prediction = 0.0;
        let mut normalized_squared_norm_x = 0.0;
        let inner_predict = |feat_value: f32, state: &[f32]| {
            let mut w_mx = state[W_MX];
            let mut w_xt = 0.0;

            let fabs_x = feat_value.abs();
            if fabs_x > w_mx {
                w_mx = fabs_x;
            }

            // COCOB update without sigmoid
            if state[W_MG] * w_mx > 0.0 {
                w_xt = ((self.config.alpha + state[W_WE])
                    / (state[W_MG] * w_mx * (state[W_MG] * w_mx + state[W_G2])))
                    * state[W_ZT];
            }

            prediction += w_xt * feat_value;
            if w_mx > 0.0 {
                let x_normalized = feat_value / w_mx;
                normalized_squared_norm_x += x_normalized * x_normalized;
            }
        };

        foreach_feature_with_state(ModelIndex::from(0), features, &self.weights, inner_predict);

        // todo select correct one
        self.model_states[0].normalized_sum_norm_x += normalized_squared_norm_x * weight;
        self.model_states[0].total_weight += weight;
        self.average_squared_norm_x =
            (self.model_states[0].normalized_sum_norm_x + 1e-6) / self.model_states[0].total_weight;

        let partial_prediction = prediction / self.average_squared_norm_x;

        // todo check nan
        partial_prediction.clamp(self.min_label, self.max_label)
    }

    fn coin_betting_update_after_predict(
        &mut self,
        features: &SparseFeatures,
        prediction: f32,
        label: f32,
        weight: f32,
    ) {
        let update =
            self.loss_function
                .first_derivative(self.min_label, self.max_label, prediction, label)
                * weight;

        let inner_update = |feat_value: f32, state: &mut [f32]| {
            //   float fabs_x = std::fabs(x);
            let fabs_x = feat_value.abs();
            let gradient = update * feat_value;
            if fabs_x > state[W_MX] {
                state[W_MX] = fabs_x;
            }
            let fabs_gradient = gradient.abs();
            if fabs_gradient > state[W_MG] {
                state[W_MG] = if fabs_gradient > self.config.beta {
                    fabs_gradient
                } else {
                    self.config.beta
                };
            }
            if state[W_MG] * state[W_MX] > 0.0 {
                state[W_XT] = ((self.config.alpha + state[W_WE])
                    / (state[W_MG] * state[W_MX] * (state[W_MG] * state[W_MX] + state[W_G2])))
                    * state[W_ZT];
            } else {
                state[W_XT] = 0.0;
            }

            state[W_ZT] += -gradient;
            state[W_G2] += gradient.abs();
            state[W_WE] += -gradient * state[W_XT];

            state[W_XT] /= self.average_squared_norm_x;
        };
        foreach_feature_with_state_mut(
            ModelIndex::from(0),
            features,
            &mut self.weights,
            inner_update,
        );
    }
}

// tests
#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

    use crate::sparse_namespaced_features::Namespace;

    use super::*;

    #[test]
    fn test_coin_betting_predict() {
        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(4);
        let coin = CoinRegressor::new(coin_config, &global_config, ModelIndex::from(1)).unwrap();
        let mut features = SparseFeatures::new();
        let ns = features.get_or_create_namespace(Namespace::Default);
        ns.add_feature(0.into(), 1.0);

        let features = Features::SparseSimple(features);

        let mut depth_info = DepthInfo::new();
        let prediction = coin.predict(&features, &mut depth_info);
        // Ensure the prediction is of variant Scalar
        assert!(matches!(prediction, Prediction::Scalar { .. }));
    }

    #[test]
    fn test_learning() {
        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(2);
        let mut coin =
            CoinRegressor::new(coin_config, &global_config, ModelIndex::from(1)).unwrap();

        let mut features = SparseFeatures::new();

        {
            let ns = features.get_or_create_namespace(Namespace::Default);
            ns.add_feature(0.into(), 1.0);
            ns.add_feature(1.into(), 1.0);
            ns.add_feature(2.into(), 1.0);
            ns.add_feature(3.into(), 1.0);
        }

        let mut depth_info = DepthInfo::new();
        let features = Features::SparseSimple(features);
        coin.learn(
            &features,
            &Label::Simple(SimpleLabel(0.5, 1.0)),
            &mut depth_info,
            0.into(),
        );
        coin.learn(
            &features,
            &Label::Simple(SimpleLabel(0.5, 1.0)),
            &mut depth_info,
            0.into(),
        );
        coin.learn(
            &features,
            &Label::Simple(SimpleLabel(0.5, 1.0)),
            &mut depth_info,
            0.into(),
        );
        coin.learn(
            &features,
            &Label::Simple(SimpleLabel(0.5, 1.0)),
            &mut depth_info,
            0.into(),
        );

        let pred = coin.predict(&features, &mut depth_info);

        assert!(matches!(pred, Prediction::Scalar { .. }));
        let pred1: &ScalarPrediction = pred.get_inner_ref().unwrap();
        match pred1 {
            ScalarPrediction { prediction, .. } => {
                assert_relative_eq!(*prediction, 0.5);
            }
            _ => unreachable!(),
        }
    }
}
