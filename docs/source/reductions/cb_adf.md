(CbAdf)=
# CbAdf

This reduction is a contextual bandit scorer for problems with action dependent features. This means that instead of there being a set number of actions as with traditional contextual bandits, there can be any number of actions and those actions are defined by the set of features which describe them. Additionally, because of this structure actions can be added or removed over time.

In order to learn a policy which can structure the relationship between action features and shared/context features interactions must be used. For example, if there is a single shared namespace called `context` and a single action namespace called `action` the following should be specified in the `globalConfig`:

```json
"interactions": [
    [{"Name": "context"}, {"Name": "action"}]
]
```

The features for this setup might look like:
```json
{
  "shared": {
    "context": {
      "my_context_feature1": 1.0,
      "my_context_feature2": 1.0
    }
  },
  "actions": [
    {
      "action": {
        "my_action_feature1": 1.0,
        "my_action_feature2": 1.0
      }
    },
    {
      "action": {
        "my_action_feature1": 1.0,
        "my_action_feature3": 1.0
      }
    }
  ]
}
```

## Configuration

```{reduction_config} CbAdf
```

## Types

- Expects: {class}`~reductionml.CbLabel`
- Expects: {class}`~reductionml.CbAdfFeatures`
- Produces: {class}`~reductionml.ActionScoresPred`
