(ElementwiseInteraction)=
# ElementwiseInteraction

This reduction provides a way to generate features as the elementwise multiplication of two vectors. The specified pair of namespaces must both contain dense features.

```{warning} This is a work in progress and the API will change.
```

To generate for multiple pairs, this reduction can be added multiple times.

## Configuration

```{reduction_config} ElementwiseInteraction
```

## Example

For the given input for a CB problem:
```json
{
  "label": { "action": 1, "cost": 1, "probability": 0.7 },
  "shared": { "shared_vector": [1.0, 2.0, 3.0] },
  "actions": [
    { "action_vector": [1.0, 2.0, 3.0] },
    { "action_vector": [4.0, 5.0, 6.0] }
  ]
}
```

An elementwise interaction could be generated with the following config:
```json
{
  "$schema": "./schemas/config/latest/schema.json",
  "entryReduction": {
    "config": {
      "cbType": "mtr",
      "regressor": {
        "config": {
          "keepOriginalFeatures": false,
          "one": { "Name": "action_vector" },
          "two": { "Name": "shared_vector" }
        },
        "typename": "ElementwiseInteraction"
      }
    },
    "typename": "CbAdf"
  },
  "globalConfig": {
    "constantFeatureEnabled": false
  }
}
```

This will result in a model which contains 3 weights, being the elementwise product of the shared and action vectors.

## Types

- Label is inherited from the base
- Expects: {class}`~reductionml.SparseFeatures`
- Prediction is inherited from the base