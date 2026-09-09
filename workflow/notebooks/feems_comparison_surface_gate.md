---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: "0.13"
    jupytext_version: "1.19.5"
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# DREEMS surface-gate comparison

This notebook compares the core DREEMS estimator, the learned null-surface
gate, and FEEMS on the exact simulations and sampling masks from
`feems_scenarios.ipynb`. All methods use the same graph and filtered SNP data.
FEEMS results are the cached leave-one-observed-deme-out fits.

```{code-cell} ipython3
from pathlib import Path

import numpy as np
from IPython.display import display

from dreems.surface_gate_comparison import (
    SCENARIO_TRUTHS,
    scenario_aggregate_figure,
    scenario_family_figure,
)
from dreems.workflow_io import read_npz

BUNDLE_PATH = Path("__SCENARIO_BUNDLE__")
bundle = read_npz(BUNDLE_PATH)

print(f"Comparison bundle: {BUNDLE_PATH}")
print(f"Core checkpoint:  {bundle['core_checkpoint_sha256'].item()}")
print(f"Gated checkpoint: {bundle['gated_checkpoint_sha256'].item()}")
print(f"FEEMS protocol:   {bundle['feems_protocol'].item()}")
```

## Aggregate comparison

For barrier and anisotropy cases, correlation measures recovery of the spatial
pattern. For homogeneous cases, correlation is undefined; output RMS and edge
MSE measure spurious surface amplitude.

```{code-cell} ipython3
scenario_aggregate_figure(bundle)
```

## Raw edge estimates

Each row uses one shared color scale. White circles mark sampled graph nodes.
The reported `alpha` projects the gated MAP estimate onto the core MAP estimate.
It measures their relative amplitude along the core pattern; MAP can also change
the pattern, so `alpha = 1` does not imply identical surfaces.

```{code-cell} ipython3
for truth_name in SCENARIO_TRUTHS:
    display(scenario_family_figure(bundle, truth_name))
```

## Per-case metrics

```{code-cell} ipython3
header = (
    "truth", "mask", "alpha",
    "core MSE", "gated MSE", "FEEMS MSE",
    "core r", "gated r", "FEEMS r",
)
print(" | ".join(header))
print(" | ".join("---" for _ in header))
for index in range(bundle["index"].size):
    values = (
        str(bundle["truth_name"][index]),
        str(bundle["sample_design"][index]),
        f"{bundle['gate_alpha_map'][index]:.3f}",
        f"{bundle['core_mse'][index]:.3f}",
        f"{bundle['gated_mse'][index]:.3f}",
        f"{bundle['feems_mse'][index]:.3f}",
        f"{bundle['core_corr'][index]:.3f}",
        f"{bundle['gated_corr'][index]:.3f}",
        f"{bundle['feems_corr'][index]:.3f}",
    )
    print(" | ".join(values))
```
