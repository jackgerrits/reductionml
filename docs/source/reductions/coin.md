
(Coin)=
# Coin

Coin is a regressor, which is based on [this paper](https://arxiv.org/abs/1602.04128). It is parameter free in that it does not require the learning rate to be specified.

## Configuration

```{reduction_config} Coin
```

## Types

- Expects: {class}`~reductionml.SimpleLabel`
- Expects: {class}`~reductionml.SparseFeatures`
- Produces: {class}`~reductionml.ScalarPred`
