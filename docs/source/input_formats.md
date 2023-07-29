# Input Formats

- [Json (recommended)](#json-format)
- Vowpal Wabbit text
  - This is the text format that Vowpal Wabbit uses. See the [docs on the VW wiki](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format) to learn how to use it.
- DsJson
  - This is the DsJson format that is offered by Vowpal Wabbit. This is provided for compatibility.

## Json format

This is a newline delimited ([`ndjson`](http://ndjson.org/)) format. Each line of input is one example to be provided to the reduction stack. This means each complete json input object json needs to be flattened to a single line.

The exact format to be used depends on the type of input being provided (features and label), which is determined based on the reductions being used.

In each format the `label` property is optional. If not provided, the input can only be used for predictions and not training.

### Feature format

Each format uses the same feature definition structure. It is an object where the keys are namespaces and the values are the contents of each namespace.

Namespace contents can be one of the following:
- List of floats
- List of strings
- Object where the keys are feature names and the values are one of the following:
  - Float
  - String
  - Bool

#### Example
```json
{
  "my_namespace": {
    "feature1": 1.0,
    "feature2": "value",
    "feature3": true
  },
  "my_namespace2": [1.0, 2.0, 3.0],
  "my_namespace3": ["value1", "value2", "value3"]
}
```

**Note**: the namespace name `:default` can be used to correspond to the Default namespace.

### Variants

Currently there are two variants which share common structures.

- [Simple](#simple) format
- [CB](#cb) format

#### Simple

Simple input is for regression scenarios.

This format produces a [`SimpleLabel`](https://docs.rs/reductionml-core/latest/reductionml_core/types/struct.SimpleLabel.html) and [`SparseFeatures`](https://docs.rs/reductionml-core/latest/reductionml_core/sparse_namespaced_features/struct.SparseFeatures.html).


```json
{
  "label": {
    "value": "float",
    "weight": "float"
  },
  "features": "<feature format>"
}
```

- If the `weight` property is not provided, it is assumed to be 1.0.
- `label` is optional. If not provided, the input can only be used for predictions and not training.

#### CB

CB input is for contextual bandit scenarios.

This format produces a [`CBLabel`](https://docs.rs/reductionml-core/latest/reductionml_core/types/struct.CBLabel.html) and [`CBAdfFeatures`](https://docs.rs/reductionml-core/latest/reductionml_core/types/struct.CBAdfFeatures.html).

```json
{
  "label": {
    "action": "integer",
    "cost": "float",
    "probability": "float"
  },
  "shared": "<feature format>",
  "actions": [
    "<feature format>",
    "..."
  ]
}
```

- `action` in the label is 0 indexed.
- `label` is optional. If not provided, the input can only be used for predictions and not training.
- `shared` is optional. If not provided, there are no shared features to be used.
