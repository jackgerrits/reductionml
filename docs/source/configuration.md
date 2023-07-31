# Configuration

A model is defined by a json configuration file. There are two sections in this file: `globalConfig` and `entryReduction`. The `globalConfig` section defines the global parameters of the model, while the `entryReduction` defines the configuration of the reductions.

The simplest configuration, to use the {doc}`reductions/coin` linear regressor would be the following:

```json
{
  "entryReduction": {
    "typename": "Coin"
  },
  "globalConfig": {}
}
```

## `entryReduction`

The entry reduction is the configuration of the topmost reduction in the stack that is being created. This reduction determines the label, feature and prediction types of the model. All reduction configuration follows a consistent structure, whether it is for the `entryReduction` or a nested reduction within.

### Reduction configuration

All reductions are configured with an an object with two properties, `typename` and `config`. Since the stack is recursive by nature, a reduction may contain one or more reduction within itself which can be configured.

- `typename` refers to the well known name of the reduction. This is listed on the reduction documentation page. For example, for {doc}`Square CB exploration <reductions/cb_explore_adf_square_cb>` the `typename` is `CbExploreAdfSquareCb`
- `config` is the configuration of the given reduction as an object. If the default configuration is sufficient, this can be omitted. The available configuration values are also listed on each reduction documentation page.

## `globalConfig`

The `globalConfig` is values that are shared across all reductions in the stack.

The global config is an object with the following properties:

- `numBits` - The number of bits to use for hashing features. The default value is 18.
- `hashSeed` - The seed to use for hashing features. The default value is 0.
- `constantFeatureEnabled` - Whether to add a constant feature to the feature vector. The default value is true.
- `interactions` - A list of interactions to use. See the [interactions](#interactions) section for more details.

## Interactions

Interactions are specified in the `globalConfig`. For example, the following is a single quadratic interaction between the default namespace and "my_namespace":

```json
"interactions": [
    ["Default", {"Name": "my_namespace"}]
]
```
