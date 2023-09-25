# Reductions

The concept of a reduction is to take a complex problem and reduce it to a simpler problem. In ReductionML these are atomic units of functionality that are typed based on the features, label and prediction types they consume and produce. Consume means the type that is passed to it and produce is the type is passes to the next reduction in the chain.

For example, contextual bandit exploration algorithms reduce to contextual bandit scorers which in turn use regression.

```{toctree}
:hidden:

cb_adf
cb_explore_adf_greedy
cb_explore_adf_softmax
cb_explore_adf_square_cb
elementwise_interaction
coin
```

## Kinds

### Linear regressors
````{card-carousel} 2

```{card} Coin
:link: coin
:link-type: doc

Parameter free linear regressor

{bdg-info}`base reduction`

```
````

### Contextual bandit exploration

````{card-carousel} 2

```{card} Epsilon Greedy
:link: cb_explore_adf_greedy
:link-type: doc

Explore based on the {math}`epsilon` value

{bdg-info}`cb exploration`

```

```{card} SquareCB
:link: cb_explore_adf_square_cb
:link-type: doc

SquareCB exploration algorithm

{bdg-info}`cb exploration`

```

```{card} Softmax
:link: cb_explore_adf_softmax
:link-type: doc

Explore based on a softmax distribution of the predicted action scores

{bdg-info}`cb exploration`

```
````

### Contextual bandit scorer

````{card-carousel} 2

```{card} CB ADF
:link: cb_adf
:link-type: doc

Contextual bandit scorer for action dependent features

```

````

### Other
````{card-carousel} 2

```{card} Elementwise Interaction
:link: elementwise_interaction
:link-type: doc

Generate features as the elementwise multiplication of two vectors

```
````
