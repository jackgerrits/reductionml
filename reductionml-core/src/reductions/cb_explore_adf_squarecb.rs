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

fn enforce_min_prob(
    uniform_epsilon: f32,
    consider_zero_valued_elements: bool,
    elements: &mut [(usize, f32)],
) -> Result<()> {
    if elements.len() == 0 {
        return Err(crate::error::Error::InvalidArgument(
            "elements.len() == 0".to_string(),
        ));
    }

    if uniform_epsilon == 0.0 {
        return Ok(());
    }

    if uniform_epsilon < 0.0 || uniform_epsilon > 1.0 {
        return Err(crate::error::Error::InvalidArgument(format!(
            "uniform_epsilon must be in [0, 1], but is {}",
            uniform_epsilon
        )));
    }

    let num_actions = elements.len();
    let support_size = if consider_zero_valued_elements {
        num_actions
    } else {
        num_actions - elements.iter().filter(|(_, p)| *p == 0.0).count()
    };

    if uniform_epsilon > 0.999 {
        elements.iter_mut().for_each(|(_, p)| {
            if consider_zero_valued_elements || *p > 0.0 {
                *p = 1.0 / support_size as f32;
            }
        });

        return Ok(());
    }

    let minimum_probability = uniform_epsilon / support_size as f32;
    let mut elements_copy = elements.to_vec();
    // Descending order. Args flipped
    elements_copy.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(p1).unwrap());

    let mut idx = 0;
    let mut running_sum = 0.0;
    let mut rho_idx = 0;
    let mut rho_sum = elements_copy[0].1;

    for (_, prob) in elements_copy {
        if !consider_zero_valued_elements && prob == 0.0 {
            break;
        }
        running_sum += prob;
        if prob
            > ((support_size - idx - 1) as f32 * minimum_probability + running_sum - 1.0)
                / (idx as f32 + 1.0)
                + minimum_probability
        {
            rho_idx = idx;
            rho_sum = running_sum;
        }

        idx += 1;
    }

    let tau = ((support_size as f32 - rho_idx as f32 - 1.0) * minimum_probability + rho_sum - 1.0)
        / (rho_idx as f32 + 1.0);
    elements.iter_mut().for_each(|(_, p)| {
        if consider_zero_valued_elements || *p > 0.0 {
            *p = (*p - tau).max(minimum_probability);
        }
    });

    Ok(())
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
        let mut cb_label_clone: CBLabel = cb_label.clone();
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::enforce_min_prob;

    #[test]
    fn test_enforce_minimum_probability() {
        let mut input = vec![(0, 1.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_no_zeros() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0)];
        enforce_min_prob(0.6, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.7, 0.3, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_all_zeros_and_dont_consider() {
        let mut input = vec![(0, 0.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.6, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.0, 0.0, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_all_zeros_and_consider() {
        let mut input = vec![(0, 0.0), (0, 0.0), (0, 0.0)];
        enforce_min_prob(0.6, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].as_slice()
        );
    }

    #[test]
    fn test_enforce_minimum_probability_equal_to_amt() {
        let mut input = vec![(0, 0.0), (0, 2.0 / 3.0), (0, 1.0 / 3.0)];
        enforce_min_prob(1.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].as_slice()
        );
    }
    #[test]
    fn test_enforce_minimum_probability_uniform() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0), (0, 0.0)];
        enforce_min_prob(1.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(
            just_probs.as_slice(),
            vec![0.25, 0.25, 0.25, 0.25].as_slice()
        );
    }

    #[test]
    #[should_panic]
    fn test_enforce_minimum_probability_bad_range() {
        enforce_min_prob(1.0, false, &mut vec![]).unwrap();
    }
    #[test]
    fn test_enforce_minimum_probability_uniform1() {
        let mut input = vec![(0, 0.9), (0, 0.1), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_uniform2() {
        let mut input = vec![(0, 0.8), (0, 0.1), (0, 0.1)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_uniform_unsorted() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.1), (0, 0.8), (0, 0.1)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.1, 0.8, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_bug_incl_zero() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.3, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.8, 0.1, 0.1].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_zero_epsilon_dont_consider() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.0, false, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.89, 0.11, 0.0].as_slice());
    }

    #[test]
    fn test_enforce_minimum_probability_zero_epsilon_consider() {
        let mut input: Vec<(usize, f32)> = vec![(0, 0.89), (0, 0.11), (0, 0.0)];
        enforce_min_prob(0.0, true, &mut input).unwrap();
        let just_probs = input.iter().map(|(_, p)| *p).collect::<Vec<_>>();
        assert_abs_diff_eq!(just_probs.as_slice(), vec![0.89, 0.11, 0.0].as_slice());
    }
}
