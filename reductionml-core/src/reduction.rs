use approx::assert_abs_diff_eq;
use serde::{Deserialize, Serialize};

use crate::{reduction_factory::PascalCaseString, types::*, ModelIndex};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DepthInfo {
    offset: ModelIndex,
}

impl DepthInfo {
    pub fn new() -> DepthInfo {
        DepthInfo { offset: 0.into() }
    }
    pub(crate) fn increment(&mut self, num_models_below: ModelIndex, i: ModelIndex) {
        self.offset = (*self.offset + (*num_models_below * *i)).into();
    }
    pub(crate) fn decrement(&mut self, num_models_below: ModelIndex, i: ModelIndex) {
        self.offset = (*self.offset - (*num_models_below * *i)).into();
    }

    pub(crate) fn absolute_offset(&self) -> ModelIndex {
        self.offset
    }
}

impl ReductionWrapper {
    pub fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        // TODO assert prediction matches expected type.
        if cfg!(debug_assertions) {
            let mut features_copy = features.clone();
            depth_info.increment(self.num_models_below, model_offset);
            let res = self.reduction.predict(features, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);

            // This is an important check to ensure that a reduction put the features back in the
            // same state as it found them.
            assert_abs_diff_eq!(features, &mut features_copy);
            res
        } else {
            depth_info.increment(self.num_models_below, model_offset);
            let res = self.reduction.predict(features, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);
            res
        }
    }
    pub fn predict_then_learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        if cfg!(debug_assertions) {
            let mut features_copy = features.clone();

            depth_info.increment(self.num_models_below, model_offset);
            let res = self
                .reduction
                .predict_then_learn(features, label, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);
            // This is an important check to ensure that a reduction put the features back in the
            // same state as it found them.
            assert_abs_diff_eq!(features, &mut features_copy);
            res
        } else {
            depth_info.increment(self.num_models_below, model_offset);
            let res = self
                .reduction
                .predict_then_learn(features, label, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);
            res
        }
    }
    pub fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) {
        // TODO assert label matches expected type.

        if cfg!(debug_assertions) {
            let mut features_copy = features.clone();

            depth_info.increment(self.num_models_below, model_offset);
            self.reduction
                .learn(features, label, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);
            // This is an important check to ensure that a reduction put the features back in the
            // same state as it found them.
            assert_abs_diff_eq!(features, &mut features_copy);
        } else {
            depth_info.increment(self.num_models_below, model_offset);
            self.reduction
                .learn(features, label, depth_info, model_offset);
            depth_info.decrement(self.num_models_below, model_offset);
        }
    }

    pub fn children(&self) -> Vec<&ReductionWrapper> {
        self.reduction.children()
    }

    // TODO work out how to handle model offset for sensitivity...
    pub fn sensitivity(
        &self,
        features: &Features,
        label: f32,
        prediction: f32,
        weight: f32,
        depth_info: DepthInfo,
    ) -> f32 {
        self.reduction
            .sensitivity(features, label, prediction, weight, depth_info)
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReductionTypeDescription {
    input_label_type: LabelType,
    output_label_type: Option<LabelType>,
    input_features_type: FeaturesType,
    output_features_type: Option<FeaturesType>,
    input_prediction_type: Option<PredictionType>,
    output_prediction_type: PredictionType,
}

impl ReductionTypeDescription {
    pub fn input_label_type(&self) -> LabelType {
        self.input_label_type
    }
    pub fn output_label_type(&self) -> Option<LabelType> {
        self.output_label_type
    }
    pub fn input_features_type(&self) -> FeaturesType {
        self.input_features_type
    }
    pub fn output_features_type(&self) -> Option<FeaturesType> {
        self.output_features_type
    }
    pub fn input_prediction_type(&self) -> Option<PredictionType> {
        self.input_prediction_type
    }
    pub fn output_prediction_type(&self) -> PredictionType {
        self.output_prediction_type
    }
}
pub struct ReductionTypeDescriptionBuilder {
    types: ReductionTypeDescription,
}

impl ReductionTypeDescriptionBuilder {
    pub fn new(
        input_label_type: LabelType,
        input_features_type: FeaturesType,
        output_prediction_type: PredictionType,
    ) -> ReductionTypeDescriptionBuilder {
        ReductionTypeDescriptionBuilder {
            types: ReductionTypeDescription::new(
                input_label_type,
                None,
                input_features_type,
                None,
                None,
                output_prediction_type,
            ),
        }
    }

    pub fn with_output_label_type(
        mut self,
        output_label_type: LabelType,
    ) -> ReductionTypeDescriptionBuilder {
        self.types.output_label_type = Some(output_label_type);
        self
    }

    pub fn with_output_features_type(
        mut self,
        output_features_type: FeaturesType,
    ) -> ReductionTypeDescriptionBuilder {
        self.types.output_features_type = Some(output_features_type);
        self
    }

    pub fn with_input_prediction_type(
        mut self,
        input_prediction_type: PredictionType,
    ) -> ReductionTypeDescriptionBuilder {
        self.types.input_prediction_type = Some(input_prediction_type);
        self
    }

    pub fn build(self) -> ReductionTypeDescription {
        self.types
    }
}

impl ReductionTypeDescription {
    fn new(
        input_label_type: LabelType,
        output_label_type: Option<LabelType>,
        input_features_type: FeaturesType,
        output_features_type: Option<FeaturesType>,
        input_prediction_type: Option<PredictionType>,
        output_prediction_type: PredictionType,
    ) -> ReductionTypeDescription {
        ReductionTypeDescription {
            input_label_type,
            output_label_type,
            input_features_type,
            output_features_type,
            input_prediction_type,
            output_prediction_type,
        }
    }

    pub fn check_and_get_reason(&self, base: &ReductionTypeDescription) -> Option<String> {
        let mut res = None;
        if self.output_label_type != Some(base.input_label_type) {
            res = Some(format!(
                "input_label_type: {:?} != {:?}",
                self.input_label_type, base.input_label_type
            ));
        }

        if self.output_features_type != Some(base.input_features_type) {
            res = Some(format!(
                "output_features_type: {:?} != {:?}",
                self.output_features_type, base.output_features_type
            ));
        }

        if self.input_prediction_type != Some(base.output_prediction_type) {
            res = Some(format!(
                "output_prediction_type: {:?} != {:?}",
                self.output_prediction_type, base.output_prediction_type
            ));
        }
        res
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReductionWrapper {
    typename: PascalCaseString,
    reduction: Box<dyn ReductionImpl>,
    type_description: ReductionTypeDescription,
    num_models_below: ModelIndex,
}

impl ReductionWrapper {
    pub fn new(
        typename: PascalCaseString,
        reduction: Box<dyn ReductionImpl>,
        type_description: ReductionTypeDescription,
        num_models_below: ModelIndex,
    ) -> ReductionWrapper {
        ReductionWrapper {
            typename,
            reduction,
            type_description,
            num_models_below,
        }
    }

    pub fn types(&self) -> &ReductionTypeDescription {
        &self.type_description
    }

    pub fn typename(&self) -> &str {
        self.typename.as_ref()
    }
}

#[typetag::serde(tag = "type")]
pub trait ReductionImpl: Send {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction;
    fn predict_then_learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    ) -> Prediction {
        let depth_info_copy: DepthInfo = *depth_info;
        let prediction = self.predict(features, depth_info, model_offset);
        let depth_info_copy2: DepthInfo = depth_info_copy;
        self.learn(features, label, depth_info, model_offset);
        assert!(depth_info == &depth_info_copy2);
        assert!(depth_info == &depth_info_copy);
        prediction
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        model_offset: ModelIndex,
    );
    fn sensitivity(
        &self,
        features: &Features,
        label: f32,
        prediction: f32,
        weight: f32,
        depth_info: DepthInfo,
    ) -> f32 {
        self.children()
            .first()
            .unwrap()
            .sensitivity(features, label, prediction, weight, depth_info)
    }
    fn children(&self) -> Vec<&ReductionWrapper>;
}
