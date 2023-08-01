(CbExploreAdfGreedy)=
# CbExploreAdfGreedy

With an epsilon greedy exploration policy the best action will be chosen with probability {math}`1 - epsilon`, and then {math}`epsilon` will be equality distributed to all actions.

In practice, this means that even if the optimal action is presented every time, it will only be selected with probability {math}`(1 - epsilon) + (epsilon/numActions)`.

So, if {math}`epsilon` is {math}`0.2` there are 5 actions being presented and the optimal action is the 0th one, the resulting probability distribution would be: {math}`[0.84, 0.04, 0.04, 0.04, 0.04]`

## Configuration

```{reduction_config} CbExploreAdfGreedy
```

## Types

- Expects: {class}`~reductionml.CbLabel`
- Expects: {class}`~reductionml.CbAdfFeatures`
- Produces: {class}`~reductionml.ActionProbsPred`
