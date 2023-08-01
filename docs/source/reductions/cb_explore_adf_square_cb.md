(CbExploreAdfSquareCb)=
# CbExploreAdfSquareCb

This reduction implements the SquareCB exploration algorithm described in [this paper (Foster and Rakhlin (2020))](https://arxiv.org/abs/2002.04926)

## Configuration

```{reduction_config} CbExploreAdfSquareCb
```

## Types

- Expects: {class}`~reductionml.CbLabel`
- Expects: {class}`~reductionml.CbAdfFeatures`
- Produces: {class}`~reductionml.ActionProbsPred`