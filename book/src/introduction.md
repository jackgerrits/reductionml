# Introduction

The purpose of most this book is mostly to cover general topics that should span across different forms of the ReductionML library. (CLI, Python, WASM, etc.) However, there will be variant specific pieces too.

## Concepts

### Reduction

The concept of a reduction is to take a complex problem and reduce it to a simpler problem. In ReductionML these are atomic units of functionality that are typed based on the features, label and prediction types they consume and produce. Consume means the type that is passed to it and produce is the type is passes to the next reduction in the chain.

For example, contextual bandit exploration algorithms reduce to contextual bandit scorers which in turn use regression.
