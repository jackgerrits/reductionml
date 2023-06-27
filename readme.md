# ReductionML

**Disclaimer**: ReductionML is my own personal project and experiment. ReductionML is an experiment and *very much a work in progress*. It is an opportunity for me to explore my own design ideas. Things are very rough and I am sure will change a lot.

ReductionML is a machine learning toolkit with solutions to a range of problems. It revolves around the concept of simplifying problems by breaking them down into more manageable components that already have solutions. This process is done by reductions as they reduce one problem to another. This approach draws inspiration from the VowpalWabbit, a project I hold in high regard and deeply value. In fact, if you are familiar with VowpalWabbit then you should be able to pick up ReductionML with ease.

## Getting started

Install:
```sh
cargo install reductionml-cli
```

There is built in support for CLI completions. To enable them, run the following commands:
### Bash (optional)

```sh
reml gen-completions bash > /usr/share/bash-completion/completions/reml
```

### Fish (optional)

```fish
reml gen-completions fish > ~/.config/fish/completions/reml.fish
```

## First steps

The following will fetch a small 100 example dataset in VW text format. It will then create a new configuration file for the Coin reduction with defaults. Finally, it will train a model using the configuration file and the dataset. The training run will be evaluated using the mean squared error metric.

```sh
curl https://raw.githubusercontent.com/VowpalWabbit/vowpal_wabbit/master/test/test-sets/0001.dat > rcv1_small.vwtxt
reml config new Coin > config.json
reml train --config config.json --data rcv1_small.vwtxt --metrics mse
```

Output:
```
warning: This CLI tool is not stable
info: Reading data file: rcv1_small.vwtxt
info: Using entry reduction: Coin
+-----------------------------+
| Example #  MeanSquaredError |
+=============================+
| 0          0                |
| 1          0.5              |
| 2          0.33609757       |
| 4          0.20314142       |
| 8          0.11399492       |
| 16         0.19589658       |
| 32         0.14362814       |
| 64         0.15037861       |
| 100        0.13510321       |
+-----------------------------+
```
