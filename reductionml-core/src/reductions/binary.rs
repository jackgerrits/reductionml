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

#[derive(Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
struct BinaryReductionConfig {
    #[serde(default = "default_regressor")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    regressor: JsonReductionConfig,
}

fn default_regressor() -> JsonReductionConfig {
    JsonReductionConfig::new(
        "Coin".try_into().unwrap(),
        json!(CoinRegressorConfig::default()),
    )
}

#[derive(Serialize, Deserialize)]
struct BinaryReduction {
    regressor: ReductionWrapper,
}

#[derive(Default)]
pub struct BinaryReductionFactory;

impl ReductionConfig for BinaryReductionConfig {
    fn typename(&self) -> PascalCaseString {
        "Binary".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReductionFactory for BinaryReductionFactory {
    impl_default_factory_functions!("Binary", BinaryReductionConfig);
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
            self.typename(),
            Box::new(BinaryReduction { regressor }),
            types,
            num_models_above,
        ))
    }
}

impl From<BinaryLabel> for SimpleLabel {
    fn from(label: BinaryLabel) -> Self {
        if label.0 { 1.0 } else { -1.0 }.into()
    }
}

#[typetag::serde]
impl ReductionImpl for BinaryReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let pred = self.regressor.predict(features, depth_info, 0.into());
        let scalar_pred: &ScalarPrediction = pred.as_inner().unwrap();

        Prediction::Binary((scalar_pred.prediction > 0.0).into())
    }

    fn predict_then_learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let binary_label: &BinaryLabel = label.as_inner().unwrap();

        let pred = self.regressor.predict_then_learn(
            features,
            &SimpleLabel::from(*binary_label).into(),
            depth_info,
            0.into(),
        );

        let scalar_pred: &ScalarPrediction = pred.as_inner().unwrap();

        Prediction::Binary((scalar_pred.prediction > 0.0).into())
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let binary_label: &BinaryLabel = label.as_inner().unwrap();

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
}
