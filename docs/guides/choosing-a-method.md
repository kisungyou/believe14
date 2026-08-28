---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Choosing a method

Start from the scientific input and the output contract, not from a familiar
class name. `believe14` keeps methods in three import families while the
registry records supervision, accepted input, approach tags, computational
cost, and whether a cited out-of-sample rule exists.

```{code-cell} ipython3
from believe14 import Capability, list_estimators

all_methods = list_estimators()
assert len(all_methods) == 30

inductive = [
    info.name
    for info in all_methods
    if Capability.TRANSFORM in info.capabilities
]
transductive = [
    info.name
    for info in all_methods
    if info.family != "estimation"
    and Capability.TRANSFORM not in info.capabilities
]

print(f"public methods: {len(all_methods)}")
print(f"inductive reducers: {len(inductive)}")
print(f"transductive reducers: {len(transductive)}")
```

## A decision sequence

1. **Choose the scientific task.** Use `estimation` for an intrinsic dimension,
   `linear` when a global map or interpretable direction is the target, and
   `nonlinear` when distances, kernels, or local neighborhoods define geometry.
2. **Match supervision and input.** Labels, continuous targets, and paired views
   lead to different estimators; precomputed distances and kernels are accepted
   only where the registry advertises them.
3. **Decide whether new observations must be embedded.** A transductive method
   deliberately has no `transform`. FastMap, Kernel PCA, and Diffusion Map are
   the only nonlinear 0.1.0 methods with cited extensions.
4. **Check the failure regime.** Read the method card before fitting tied
   neighbors, a disconnected graph, rank-deficient data, or extreme scales.
5. **Read `diagnostics_`.** A successful fit reports the solver and honest
   convergence state; iterative estimators never turn `max_iter` exhaustion
   into convergence.

Registry filters compose, so discovery remains stable as future methods are
added.

```{code-cell} ipython3
graph_methods = list_estimators(family="nonlinear", approach="graph")
supervised_linear = list_estimators(family="linear", supervision="supervised")

print("graph:", ", ".join(info.name for info in graph_methods))
print("supervised linear:", ", ".join(info.name for info in supervised_linear))
```

## Executable coverage

This table is generated from the immutable public registry and method-card
front matter. The documentation build fails if a card is missing, duplicated,
unexpected, or assigned to the wrong family.

```{believe14-example-coverage}
```
