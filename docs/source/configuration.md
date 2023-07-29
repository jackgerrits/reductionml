# Configuration

A model is defined by a json configuration file. There are two sections in this file: `globalConfig` and `entryReduction`. The `globalConfig` section defines the global parameters of the model, while the `entryReduction` defines the configuration of the reductions.

The simplest configuration, to use a linear regressor is the following:

```json
{
  "entryReduction": {
    "typename": "Coin"
  },
  "globalConfig": {}
}
```

## Reduction config

When defining a reduction, either for the `entryReduction` or a nested reduction definition you must specify  two properties. `typename`, which corresponds to the name of the reduction to instantiate and `config` when contains the reduction specific configuration. This config could in turn include another reduction config since they are recursive by nature.

## Interactions

Interactions are specified in the global config. For example, the following is a single quadratic interaction between the default namespace and "namespace":

```json
{
    "globalConfig": {
        "interactions": [
            ["Default", {"Nam": "namespace"}]
        ]
    }
}
```
