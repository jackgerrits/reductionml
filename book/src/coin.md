# Coin

- Typename: `Coin`

Coin is a regressor, which is based on [this paper](https://arxiv.org/abs/1602.04128). It is parameter free in that it does not require the learning rate to be specified.

## Configuration

`reduction-config-json=Coin`


## Types

- This format expects a [`SimpleLabel`](https://docs.rs/reductionml-core/0.0.1/reductionml_core/types/struct.SimpleLabel.html)
- This format expects [`SparseFeatures`](https://docs.rs/reductionml-core/0.0.1/reductionml_core/sparse_namespaced_features/struct.SparseFeatures.html)
- This format produces a [`ScalarPrediction`](https://docs.rs/reductionml-core/0.0.1/reductionml_core/types/struct.ScalarPrediction.html)