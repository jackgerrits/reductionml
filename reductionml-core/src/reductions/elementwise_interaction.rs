use crate::error::Result;
use crate::global_config::GlobalConfig;
use crate::interactions::NamespaceDef;
use crate::reduction::{
    DepthInfo, ReductionImpl, ReductionTypeDescriptionBuilder, ReductionWrapper,
};
use crate::reduction_factory::{
    create_reduction, JsonReductionConfig, PascalCaseString, ReductionConfig, ReductionFactory,
};
use crate::sparse_namespaced_features::{Namespace, SparseFeatures, SparseFeaturesNamespace};
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
struct ElementwiseInteractionConfig {
    #[serde(default = "default_regressor")]
    #[schemars(schema_with = "crate::config_schema::gen_json_reduction_config_schema")]
    regressor: JsonReductionConfig,

    /// TODO: document that is it pairs for now. But could conceptually be more.
    /// FIXME: this default doesn't really make sense.
    #[serde(default = "default_ns")]
    one: NamespaceDef,
    #[serde(default = "default_ns")]
    two: NamespaceDef,

    /// Default is false.
    #[serde(default)]
    keep_original_features: bool,
}

fn default_regressor() -> JsonReductionConfig {
    JsonReductionConfig::new(
        "Coin".try_into().unwrap(),
        json!(CoinRegressorConfig::default()),
    )
}

fn default_ns() -> NamespaceDef {
    NamespaceDef::Default
}

#[derive(Serialize, Deserialize)]
struct ElementwiseInteraction {
    regressor: ReductionWrapper,
    pair: (Namespace, Namespace),
    keep_original_features: bool,
    num_bits: u8,
}

#[derive(Default)]
pub struct ElementwiseInteractionFactory;

impl ReductionConfig for ElementwiseInteractionConfig {
    fn typename(&self) -> PascalCaseString {
        "ElementwiseInteraction".try_into().unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReductionFactory for ElementwiseInteractionFactory {
    impl_default_factory_functions!("ElementwiseInteraction", ElementwiseInteractionConfig);
    fn create(
        &self,
        config: &dyn ReductionConfig,
        global_config: &GlobalConfig,
        num_models_above: ModelIndex,
    ) -> Result<ReductionWrapper> {
        let config = config
            .as_any()
            .downcast_ref::<ElementwiseInteractionConfig>()
            .unwrap();
        let regressor_config = crate::reduction_factory::parse_config(&config.regressor)?;
        let regressor: ReductionWrapper =
            create_reduction(regressor_config.as_ref(), global_config, num_models_above)?;

        let types = ReductionTypeDescriptionBuilder::new(
            regressor.types().input_label_type(),
            FeaturesType::SparseSimple,
            regressor.types().output_prediction_type(),
        )
        .with_input_prediction_type(regressor.types().output_prediction_type())
        .with_output_features_type(FeaturesType::SparseSimple)
        .with_output_label_type(regressor.types().input_label_type())
        .build();

        if let Some(reason) = types.check_and_get_reason(regressor.types()) {
            return Err(crate::error::Error::InvalidArgument(format!(
                "Invalid reduction configuration: {}",
                reason
            )));
        }

        let (one, two) = (
            config.one.to_namespace(global_config.hash_seed()),
            config.two.to_namespace(global_config.hash_seed()),
        );

        Ok(ReductionWrapper::new(
            self.typename(),
            Box::new(ElementwiseInteraction {
                regressor,
                pair: (one, two),
                keep_original_features: config.keep_original_features,
                num_bits: global_config.num_bits(),
            }),
            types,
            num_models_above,
        ))
    }
}

fn elementwise_multiply(
    num_bits: u8,
    features: &SparseFeatures,
    one: &Namespace,
    two: &Namespace,
) -> (Namespace, SparseFeaturesNamespace) {
    let (one, two) = (one.clone(), two.clone());
    let composite_hash: u32 = (u32::from(one.hash()) as u64 * u32::from(two.hash()) as u64) as u32;
    let mask = FeatureMask::from_num_bits(num_bits);
    let base_one = FeatureHash::from(u32::from(one.hash())).mask(mask);
    let base_two = FeatureHash::from(u32::from(two.hash())).mask(mask);

    let composite_ns = Namespace::RawHash(composite_hash.into());
    let composite_base = FeatureHash::from(u32::from(composite_ns.hash())).mask(mask);

    let mut dest_iter = SparseFeaturesNamespace::new(composite_ns);
    let mut ns_one_it = features.get_namespace(one).unwrap().iter();
    let mut ns_two_it = features.get_namespace(two).unwrap().iter();
    let mut cur_1 = ns_one_it.next();
    let mut cur_2 = ns_two_it.next();
    loop {
        let (idx1, idx2, val1, val2) = match (cur_1, cur_2) {
            (None, None) => break,
            (None, Some(_)) => break,
            (Some(_), None) => break,
            (Some(a), Some(b)) => {
                let (idx1, val1) = a;
                let (idx2, val2) = b;
                (idx1, idx2, val1, val2)
            }
        };

        let idx1 = u32::from(idx1) - u32::from(base_one);
        let idx2 = u32::from(idx2) - u32::from(base_two);

        let dest_index = u32::from(composite_base) + idx1;

        if idx1 == idx2 {
            dest_iter.add_feature(dest_index.into(), val1 * val2);
            cur_1 = ns_one_it.next();
            cur_2 = ns_two_it.next();
        } else if idx1 < idx2 {
            cur_1 = ns_one_it.next();
        } else {
            cur_2 = ns_two_it.next();
        }
    }

    (composite_ns, dest_iter)
}

#[typetag::serde]
impl ReductionImpl for ElementwiseInteraction {
    fn predict(
        &self,
        features: &mut Features,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();

        let (composite_ns, dest_iter) =
            elementwise_multiply(self.num_bits, sparse_feats, &self.pair.0, &self.pair.1);

        sparse_feats.set_namespace(composite_ns, dest_iter);
        if self.keep_original_features {
            let pred = self
                .regressor
                .predict(&mut sparse_feats.into(), depth_info, 0.into());
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
            pred
        } else {
            let ns1 = sparse_feats.remove_namespace(self.pair.0).unwrap();
            let ns2 = sparse_feats.remove_namespace(self.pair.1).unwrap();
            let pred = self
                .regressor
                .predict(&mut sparse_feats.into(), depth_info, 0.into());
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
            sparse_feats.set_namespace(self.pair.0, ns1);
            sparse_feats.set_namespace(self.pair.1, ns2);
            pred
        }
    }

    fn predict_then_learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) -> Prediction {
        let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();

        let (composite_ns, dest_iter) =
            elementwise_multiply(self.num_bits, sparse_feats, &self.pair.0, &self.pair.1);

        sparse_feats.set_namespace(composite_ns, dest_iter);
        if self.keep_original_features {
            let pred = self.regressor.predict_then_learn(
                &mut sparse_feats.into(),
                label,
                depth_info,
                0.into(),
            );
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
            pred
        } else {
            let ns1 = sparse_feats.remove_namespace(self.pair.0).unwrap();
            let ns2 = sparse_feats.remove_namespace(self.pair.1).unwrap();
            let pred = self.regressor.predict_then_learn(
                &mut sparse_feats.into(),
                label,
                depth_info,
                0.into(),
            );
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
            sparse_feats.set_namespace(self.pair.0, ns1);
            sparse_feats.set_namespace(self.pair.1, ns2);
            pred
        }
    }

    fn learn(
        &mut self,
        features: &mut Features,
        label: &Label,
        depth_info: &mut DepthInfo,
        _model_offset: ModelIndex,
    ) {
        let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();

        let (composite_ns, dest_iter) =
            elementwise_multiply(self.num_bits, sparse_feats, &self.pair.0, &self.pair.1);

        sparse_feats.set_namespace(composite_ns, dest_iter);
        if self.keep_original_features {
            self.regressor
                .learn(&mut sparse_feats.into(), label, depth_info, 0.into());
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
        } else {
            let ns1 = sparse_feats.remove_namespace(self.pair.0).unwrap();
            let ns2 = sparse_feats.remove_namespace(self.pair.1).unwrap();
            self.regressor
                .learn(&mut sparse_feats.into(), label, depth_info, 0.into());
            let sparse_feats: &mut SparseFeatures = features.as_inner_mut().unwrap();
            sparse_feats.remove_namespace(composite_ns);
            sparse_feats.set_namespace(self.pair.0, ns1);
            sparse_feats.set_namespace(self.pair.1, ns2);
        }
    }

    fn children(&self) -> Vec<&ReductionWrapper> {
        vec![&self.regressor]
    }
}
