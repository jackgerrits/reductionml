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

#[derive(Deserialize, Serialize, JsonSchema, DefaultFromSerde)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct DebugConfig {
    #[serde(default = "default_cb_type")]
    id: String,
    #[serde(default = "default_false")]
    prediction: bool,
    #[serde(default = "default_false")]
    label: bool,
    #[serde(default = "default_false")]
    features: bool,
    #[serde(default = "default_indent")]
    indent: usize,
    #[serde(default = "default_next")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    next: JsonReductionConfig,
}

fn default_cb_type() -> String {
    "".to_owned()
}

fn default_false() -> bool {
    false
}

fn default_indent() -> usize {
    0
}

fn default_next() -> JsonReductionConfig {
    JsonReductionConfig::new("Unknown".try_into().unwrap(), json!({}))
}

impl ReductionConfig for DebugConfig {
    fn typename(&self) -> PascalCaseString {
        "Debug".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
struct DebugReduction {
    id: String,
    indent: usize,
    prediction: bool,
    label: bool,
    features: bool,
    next: ReductionWrapper,
}

#[derive(Default)]
pub struct DebugReductionFactory;

impl ReductionFactory for DebugReductionFactory {
    impl_default_factory_functions!("Debug", DebugConfig);

    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config.as_any().downcast_ref::<DebugConfig>().unwrap();
        let next_config = crate::reduction_factory::parse_config(&config.next)?;
        let next: ReductionWrapper =
            create_reduction(next_config.as_ref(), global_config, num_models_above)?;

        let types: crate::reduction::ReductionTypeDescription =
            ReductionTypeDescriptionBuilder::new(
                next.types().input_label_type(),
                next.types().input_features_type(),
                next.types().output_prediction_type(),
            )
            .with_output_features_type(next.types().input_features_type())
            .with_input_prediction_type(next.types().output_prediction_type())
            .with_output_label_type(next.types().input_label_type())
            .build();

        if let Some(reason) = types.check_and_get_reason(next.types()) {
            return Err(crate::error::Error::InvalidArgument(format!(
                "Invalid reduction configuration: {}",
                reason
            )));
        }

        Ok(ReductionWrapper::new(
            self.typename(),
            Box::new(DebugReduction {
                id: config.id.clone(),
                indent: config.indent,
                prediction: config.prediction,
                features: config.features,
                label: config.label,
                next,
            }),
            types,
            num_models_above,
        ))
    }
}

impl DebugReduction {
    fn print_debug<S: AsRef<str>>(
        &self,
        func: S,
        offset: ModelIndex,
        depth_info: &DepthInfo,
        msg: S,
    ) {
        let space = " ";
        let indent = self.indent;
        let id = &self.id;
        let func = func.as_ref();
        let msg = msg.as_ref();
        let off = u8::from(offset);
        let abs_off = u8::from(depth_info.absolute_offset());
        eprintln!("{space:indent$}[{id}({func}), off: {off}, abs_off: {abs_off}] {msg}");
    }
}

#[typetag::serde]
impl ReductionImpl for DebugReduction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        if self.features {
            self.print_debug(
                "predict",
                model_offset,
                depth_info,
                &format!("features: {:?}", features),
            );
        }
        let prediction = self.next.predict(features, depth_info, 0.into());

        if self.prediction {
            self.print_debug(
                "predict",
                model_offset,
                depth_info,
                &format!("prediction: {:?}", prediction),
            );
        }
        prediction
    }

    fn predict_then_learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        if self.features {
            self.print_debug(
                "predict_then_learn",
                model_offset,
                depth_info,
                &format!("features: {:?}", features),
            );
        }

        if self.label {
            self.print_debug(
                "predict_then_learn",
                model_offset,
                depth_info,
                &format!("label: {:?}", label),
            );
        }

        let prediction = self
            .next
            .predict_then_learn(features, label, depth_info, model_offset);
        if self.prediction {
            self.print_debug(
                "predict_then_learn",
                model_offset,
                depth_info,
                &format!("prediction: {:?}", prediction),
            );
        }
        prediction
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) {
        if self.features {
            self.print_debug(
                "learn",
                model_offset,
                depth_info,
                &format!("features: {:?}", features),
            );
        }

        if self.label {
            self.print_debug(
                "learn",
                model_offset,
                depth_info,
                &format!("label: {:?}", label),
            );
        }
        self.next.learn(features, label, depth_info, 0.into());
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.next]
    }
}
