# Reductions

The concept of a reduction is to take a complex problem and reduce it to a simpler problem. In ReductionML these are atomic units of functionality that are typed based on the features, label and prediction types they consume and produce. Consume means the type that is passed to it and produce is the type is passes to the next reduction in the chain.

For example, contextual bandit exploration algorithms reduce to contextual bandit scorers which in turn use regression.

```{toctree}
:hidden:

cb_adf
cb_explore_adf_greedy
cb_explore_adf_square_cb.md
coin
```

