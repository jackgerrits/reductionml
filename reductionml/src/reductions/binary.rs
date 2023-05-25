use crate::error::{Error, Result};
use crate::global_config::GlobalConfig;
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, ReductionConfig, ReductionFactory,
};
use crate::utils::GetInner;

use crate::reductions::CoinRegressorConfig;
use crate::{types::*, ModelIndex};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct BinaryReductionConfig {
    #[serde(default = "default_regressor")]
    regressor: JsonReductionConfig,
}

fn default_regressor() -> JsonReductionConfig {
    JsonReductionConfig::new("coin".to_owned(), json!(CoinRegressorConfig::default()))
}

#[derive(Serialize, Deserialize)]
struct BinaryReduction {
    regressor: ReductionWrapper,
}

#[derive(Default)]
pub struct BinaryReductionFactory;

impl ReductionConfig for BinaryReductionConfig {
    fn typename(&self) -> String {
        "binary".to_owned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReductionFactory for BinaryReductionFactory {
    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<BinaryReductionConfig>()
            .unwrap();
        let regressor_config = crate::reduction_factory::parse_config(&config.regressor)?;
        let regressor: ReductionWrapper =
            create_reduction(regressor_config.as_ref(), global_config, num_models_above)?;

        let types = ReductionTypeDescriptionBuilder::new(
            LabelType::Binary,
            regressor.types().input_features_type(),
            PredictionType::Binary,
        )
        .with_input_prediction_type(PredictionType::Scalar)
        .with_output_features_type(regressor.types().input_features_type())
        .with_output_label_type(LabelType::Simple)
        .build();

        if let Some(reason) = types.check_and_get_reason(regressor.types()) {
            return Err(crate::error::Error::InvalidArgument(format!(
                "Invalid reduction configuration: {}",
                reason
            )));
        }

        Ok(ReductionWrapper::new(
            Box::new(BinaryReduction { regressor }),
            types,
            num_models_above,
        ))
    }

    fn typename(&self) -> String {
        "binary".to_owned()
    }

    fn parse_config(
        &self,
        value: &serde_json::Value,
    ) -> Result<Box<dyn crate::reduction_factory::ReductionConfig>> {
        if value["typename"] != "binary" {
            return Err(Error::InvalidArgument(
                "Invalid typename for binary reduction".to_owned(),
            ));
        }
        let res: BinaryReductionConfig = serde_json::from_value(value.clone()).unwrap();
        Ok(Box::new(res))
    }
}

impl From<BinaryLabel> for SimpleLabel {
    fn from(label: BinaryLabel) -> Self {
        if label.0 { 1.0 } else { -1.0 }.into()
    }
}

#[typetag::serde]
impl ReductionImpl for BinaryReduction {
    fn predict(&self, features: &Features, depth_info: &mut DepthInfo) -> Prediction {
        let pred = self.regressor.predict(features, depth_info, 0.into());
        let scalar_pred: &ScalarPrediction = pred.get_inner_ref().unwrap();

        Prediction::Binary((scalar_pred.prediction > 0.0).into())
    }

    fn predict_then_learn(
        &mut self,
        features: &Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let binary_label: &BinaryLabel = label.get_inner_ref().unwrap();

        let pred = self.regressor.predict_then_learn(
            features,
            &SimpleLabel::from(*binary_label).into(),
            depth_info,
            0.into(),
        );

        let scalar_pred: &ScalarPrediction = pred.get_inner_ref().unwrap();

        Prediction::Binary((scalar_pred.prediction > 0.0).into())
    }

    fn learn(
        &mut self,
        features: &Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let binary_label: &BinaryLabel = label.get_inner_ref().unwrap();

        self.regressor.learn(
            features,
            &SimpleLabel::from(*binary_label).into(),
            depth_info,
            0.into(),
        )
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.regressor]
    }

    fn typename(&self) -> String {
        "binary".to_owned()
    }
}
