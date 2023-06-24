use std::iter::Sum;
use std::ops::Deref;

use crate::dense_weights::DenseWeights;
use crate::error::Result;
use crate::global_config::GlobalConfig;
use crate::interactions::{compile_interactions, Interaction};
use crate::loss_function::{LossFunction, LossFunctionType};
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{ReductionConfig, ReductionFactory};
use crate::sparse_namespaced_features::{Namespace, SparseFeatures};
use crate::utils::bits_to_max_feature_index;
use crate::utils::GetInner;
use crate::weights::{foreach_feature, foreach_feature_with_state, foreach_feature_with_state_mut};
use crate::{impl_default_factory_functions, types::*, ModelIndex, StateIndex};
use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;

#[derive(Deserialize, DefaultFromSerde, Serialize, Debug, Clone, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoinRegressorConfig {
    #[serde(default = "default_alpha")]
    alpha: f32,

    #[serde(default = "default_beta")]
    beta: f32,

    #[serde(default)]
    l1_lambda: f32,

    #[serde(default)]
    l2_lambda: f32,

    #[serde(default)]
    interactions: Option<Vec<Interaction>>,
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
    // TODO allow this to be chosen
    loss_function: LossFunctionHolder,
    pairs: Option<Vec<(Namespace, Namespace)>>,
    triples: Option<Vec<(Namespace, Namespace, Namespace)>>,
    num_bits: u8,
    expect_constant_feature: bool,
}

impl CoinRegressor {
    pub fn new(
        config: CoinRegressorConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<CoinRegressor> {
        let (pairs, triples) = match config
            .interactions
            .as_ref()
            .map(|x| compile_interactions(x, global_config.hash_seed()))
        {
            Some((Some(pairs), Some(triples))) => (Some(pairs), Some(triples)),
            Some((None, Some(triples))) => (None, Some(triples)),
            Some((Some(pairs), None)) => (Some(pairs), None),
            Some((None, None)) => (None, None),
            None => (None, None),
        };
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
            pairs,
            triples,
            num_bits: global_config.num_bits(),
            expect_constant_feature: global_config.add_constant_feature(),
        })
    }
}

#[derive(Default)]
pub struct CoinRegressorFactory;

impl ReductionFactory for CoinRegressorFactory {
    impl_default_factory_functions!("coin", CoinRegressorConfig);

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
            self.typename(),
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
}

#[typetag::serde]
impl ReductionImpl for CoinRegressor {
    fn predict(
        &self,
        features: &Features,
        _depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        let sparse_feats: &SparseFeatures = features.get_inner_ref().unwrap();
        assert!(
            !(self.expect_constant_feature && !sparse_feats.constant_feature_exists()),
            "Constant feature was expected but not present."
        );
        assert!(
            !(!self.expect_constant_feature && sparse_feats.constant_feature_exists()),
            "Constant feature was not expected but present."
        );

        let mut prediction = 0.0;
        foreach_feature(
            0.into(),
            sparse_feats,
            &self.weights,
            &self.pairs,
            &self.triples,
            self.num_bits,
            |feat_val, weight_val| prediction += feat_val * weight_val,
        );

        if prediction.is_nan() {
            prediction = 0.0;
        }

        let scalar_pred = ScalarPrediction {
            prediction: prediction.clamp(self.min_label, self.max_label),
            raw_prediction: prediction,
        };
        scalar_pred.into()
    }

    fn predict_then_learn(
        &mut self,
        features: &Features,
        label: &Label,
        _depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let sparse_feats: &SparseFeatures = features.get_inner_ref().unwrap();
        assert!(self.expect_constant_feature && sparse_feats.constant_feature_exists() || !self.expect_constant_feature && !sparse_feats.constant_feature_exists(), "Constant feature must be present iff add_constant_feature was passed in global config.");
        let simple_label: &SimpleLabel = label.get_inner_ref().unwrap();

        self.min_label = simple_label.0.min(self.min_label);
        self.max_label = simple_label.0.max(self.max_label);
        let prediction = self.coin_betting_predict(sparse_feats, simple_label.1);
        self.coin_betting_update_after_predict(
            sparse_feats,
            prediction,
            simple_label.0,
            simple_label.1,
        );
        let scalar_pred = ScalarPrediction {
            prediction: prediction.clamp(self.min_label, self.max_label),
            raw_prediction: prediction,
        };
        scalar_pred.into()
    }

    fn learn(
        &mut self,
        features: &Features,
        label: &Label,
        _depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let sparse_feats: &SparseFeatures = features.get_inner_ref().unwrap();
        assert!(self.expect_constant_feature && sparse_feats.constant_feature_exists() || !self.expect_constant_feature && !sparse_feats.constant_feature_exists(), "Constant feature must be present iff add_constant_feature was passed in global config.");
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
}

const W_XT: usize = 0; //  current parameter
const W_ZT: usize = 1; //  sum negative gradients
const W_G2: usize = 2; //  sum of absolute value of gradients
const W_MX: usize = 3; //  maximum absolute value
const W_WE: usize = 4; //  Wealth
const W_MG: usize = 5; //  Maximum Lipschitz constant

// TODO constant

struct PredOutcome(f32, f32);

impl Sum for PredOutcome {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = PredOutcome(0.0, 0.0);
        for PredOutcome(x, y) in iter {
            sum.0 += x;
            sum.1 += y;
        }
        sum
    }
}

impl CoinRegressor {
    fn coin_betting_predict(&mut self, features: &SparseFeatures, weight: f32) -> f32 {
        let mut prediction = 0.0;
        let mut normalized_squared_norm_x = 0.0;

        let inner_predict = |feat_value: f32, state: &[f32]| {
            assert!(state.len() == 6);

            let w_mx = state[W_MX].max(feat_value.abs());

            // COCOB update without sigmoid
            let w_xt = if state[W_MG] * w_mx > 0.0 {
                ((self.config.alpha + state[W_WE])
                    / (state[W_MG] * w_mx * (state[W_MG] * w_mx + state[W_G2])))
                    * state[W_ZT]
            } else {
                0.0
            };

            prediction += w_xt * feat_value;
            if w_mx > 0.0 {
                let x_normalized = feat_value / w_mx;
                normalized_squared_norm_x += x_normalized * x_normalized;
            } else {
            }
        };

        foreach_feature_with_state(
            0.into(),
            features,
            &self.weights,
            &self.pairs,
            &self.triples,
            self.num_bits,
            inner_predict,
        );

        // todo select correct one
        self.model_states[0].normalized_sum_norm_x += normalized_squared_norm_x * weight;
        self.model_states[0].total_weight += weight;
        self.average_squared_norm_x =
            (self.model_states[0].normalized_sum_norm_x + 1e-6) / self.model_states[0].total_weight;

        let partial_prediction = prediction / self.average_squared_norm_x;

        // dbg!(partial_prediction);

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

        // dbg!(update);

        let inner_update = |feat_value: f32, state: &mut [f32]| {
            assert!(state.len() == 6);
            // dbg!(feat_value);
            //   float fabs_x = std::fabs(x);
            let fabs_x = feat_value.abs();
            let gradient = update * feat_value;
            if fabs_x > state[W_MX] {
                state[W_MX] = fabs_x;
            }
            let fabs_gradient = update.abs();
            // if (fabs_gradient > w[W_MG]) { w[W_MG] = fabs_gradient > d.ftrl_beta ? fabs_gradient : d.ftrl_beta; }

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

            // dbg!(state[W_XT]);
            // dbg!(state[W_ZT]);
            // dbg!(state[W_G2]);
            // dbg!(state[W_MX]);
            // dbg!(state[W_WE]);
            // dbg!(state[W_MG]);
            // dbg!("---");
        };
        foreach_feature_with_state_mut(
            ModelIndex::from(0),
            features,
            &mut self.weights,
            &self.pairs,
            &self.triples,
            self.num_bits,
            inner_update,
        );
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::sparse_namespaced_features::Namespace;

    use super::*;

    #[test]
    fn test_coin_betting_predict() {
        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(4, 0, false);
        let coin = CoinRegressor::new(coin_config, &global_config, ModelIndex::from(1)).unwrap();
        let mut features = SparseFeatures::new();
        let ns = features.get_or_create_namespace(Namespace::Default);
        ns.add_feature(0.into(), 1.0);

        let features = Features::SparseSimple(features);

        let mut depth_info = DepthInfo::new();
        let prediction = coin.predict(&features, &mut depth_info, 0.into());
        // Ensure the prediction is of variant Scalar
        assert!(matches!(prediction, Prediction::Scalar { .. }));
    }

    #[test]
    fn test_learning() {
        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(2, 0, false);
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

        let pred = coin.predict(&features, &mut depth_info, 0.into());

        assert!(matches!(pred, Prediction::Scalar { .. }));
        let pred1: &ScalarPrediction = pred.get_inner_ref().unwrap();
        assert_relative_eq!(pred1.prediction, 0.5);
    }

    fn test_learning_e2e(x: fn(i32) -> f32, yhat: fn(f32) -> f32, n: i32, mut regressor: CoinRegressor) {
        for i in 0..n {
            let mut features = SparseFeatures::new();
            features.add_constant_feature(2);
            let _x = x(i); 
            {
                let ns = features.get_or_create_namespace(Namespace::Default);
                ns.add_feature(0.into(), _x);
            }
    
            let mut depth_info = DepthInfo::new();
            let features = Features::SparseSimple(features);
            regressor.learn(
                &features,
                &Label::Simple(SimpleLabel(yhat(_x), 1.0)),
                &mut depth_info,
                0.into(),
            );            
        }

        let test_set = [0.0, 1.0, 2.0, 3.0];
        for x in test_set {
            let mut features = SparseFeatures::new();
            features.add_constant_feature(2);
            {
                let ns = features.get_or_create_namespace(Namespace::Default);
                ns.add_feature(0.into(), x);
                
            }
    
            let mut depth_info = DepthInfo::new();
            let features = Features::SparseSimple(features);
            let pred = regressor.predict(&features, &mut depth_info, 0.into());
            assert!(matches!(pred, Prediction::Scalar { .. }));

            let pred_value: &ScalarPrediction = pred.get_inner_ref().unwrap();
            assert_relative_eq!(pred_value.prediction, yhat(x), epsilon=0.001);
        }
    }

     #[test]
    fn test_learning_const() {
        fn x(i: i32) -> f32 {
            (i % 100) as f32 / 10.0
        }
        fn yhat(x: f32) -> f32 { 1.0 }

        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(4, 0, true);
        let mut coin: CoinRegressor =
            CoinRegressor::new(coin_config, &global_config, ModelIndex::from(1)).unwrap();

        test_learning_e2e(x, yhat, 10000, coin);
    } 

    #[test]
    fn test_learning_linear() {
        fn x(i: i32) -> f32 {
            (i % 100) as f32 / 10.0
        }
        fn yhat(x: f32) -> f32 { 2.0 * x + 3.0 }

        let coin_config = CoinRegressorConfig::default();
        let global_config = GlobalConfig::new(4, 0, true);
        let mut coin: CoinRegressor =
            CoinRegressor::new(coin_config, &global_config, ModelIndex::from(1)).unwrap();

        test_learning_e2e(x, yhat, 10000, coin);
    } 
}
