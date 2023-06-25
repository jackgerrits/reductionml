use schemars::{
    gen::SchemaGenerator,
    schema::{RootSchema, Schema, SchemaObject},
    schema_for,
};

use crate::{reduction_factory::ReductionFactory, workspace::Configuration};

pub struct ConfigSchema {
    schema: RootSchema,
}

impl Default for ConfigSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigSchema {
    pub fn new() -> Self {
        let mut schema = schema_for!(Configuration);

        // Create global all reductions schema
        let mut any_reduction_config = SchemaObject::default();
        any_reduction_config.subschemas().one_of = Some(vec![]);

        schema.definitions.insert(
            "any_reduction_config".to_owned(),
            Schema::Object(any_reduction_config),
        );

        // Allow $schema to be set to anything
        let mut schema_schema: SchemaObject = SchemaObject::default();
        schema_schema.string().pattern = Some(".*".to_owned());
        schema
            .schema
            .object()
            .properties
            .insert("$schema".to_owned(), schema_schema.into());

        Self { schema }
    }

    pub fn schema(&self) -> &RootSchema {
        &self.schema
    }

    pub fn add_reduction(&mut self, reduction_factory: &dyn ReductionFactory) {
        let typename_constant = schemars::schema::SchemaObject {
            const_value: Some(reduction_factory.typename().as_ref().into()),
            ..Default::default()
        };

        let mut reduction_config_schema = SchemaObject::default();
        reduction_config_schema
            .object()
            .properties
            .insert("typename".to_owned(), Schema::Object(typename_constant));

        let reductions_inner_schema = reduction_factory.get_config_schema();
        reduction_config_schema.object().properties.insert(
            "config".to_owned(),
            Schema::Object(reductions_inner_schema.schema),
        );
        // Set additionalProperties to false
        reduction_config_schema.object().additional_properties =
            Some(Box::new(Schema::Bool(false)));

        self.schema.definitions.insert(
            reduction_factory.typename().as_ref().into(),
            Schema::Object(reduction_config_schema),
        );

        match self
            .schema
            .definitions
            .get_mut("any_reduction_config")
            .unwrap()
        {
            Schema::Object(any_reduction_config) => {
                let new_reduction_ref = schemars::schema::SchemaObject {
                    reference: Some(format!("#/definitions/{}", reduction_factory.typename())),
                    ..Default::default()
                };
                any_reduction_config
                    .subschemas()
                    .one_of
                    .as_mut()
                    .unwrap()
                    .push(Schema::Object(new_reduction_ref));
            }
            _ => panic!("any_reduction_config is not an object"),
        }

        // TODO: Handle duplicate definitions
        self.schema
            .definitions
            .extend(reductions_inner_schema.definitions);

        // TODO handle "true" reduction
    }
}

pub(crate) fn gen_json_reduction_config_schema(_gen: &mut SchemaGenerator) -> Schema {
    schemars::schema::SchemaObject {
        reference: Some("#/definitions/any_reduction_config".to_owned()),
        ..Default::default()
    }
    .into()
}
