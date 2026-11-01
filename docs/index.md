# Welcome to DREEMS' documentation
DREEMS - Deep Recovery of Estimated Effective Migration Surfaces - is a Python package for computing and visualizing relative effective migration surfaces. DREEMS inference function is exposed as a black-box map from covariance and geography to a migration surface. 

The entire inference procedure of DREEMS is one line of code. To understand the necessary inputs and how to visualize migration surfaces, pleasee see the tutorials.

```python
surface = dreems_infer(data=ds, sample_coordinates=sample_coordinates, nodes=nodes, edges=edges)
```

The inference pipeline accepts uniformly haploid, diploid, or polyploid allele calls. Given a LD-pruned dataset, it retains biallelic variants passing common MAF and call-rate filters, mean-imputes missing dosages, standardizes each SNP, assigns samples to their nearest graph node, computes node-level covariance, and outputs a migration surface.  The default filters are MAF >= 0.05 and variant call rate >= 0.8. Apply study-specific sample, read-depth, genotype-quality, and LD filters before conversion. This function does not perform those checks or infer an LD-adjusted SNP count.

DREEMS accepts an xarray genotype dataset using the standard sgkit dimensions and variables, please see [`Data Processing`](data_preprocessing.md). For those familiar with FEEMS and want to directly run DREEMS using FEEMS inputs, please see [`Using FEEMS inputs`](wolves.md).

## Content
```{toctree}
:maxdepth: 1
Data Preprocessing <data_preprocessing>
Visualizing Surfaces <plotting>
Blacklegged Ticks <blacklegged_ticks>
Allohistium Fishes <fishes>
Using FEEMS Inputs <wolves>
API <api/index>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

## License
The MIT License (MIT)

Copyright (c) 2026 Jiatong (Kevin) Liang and Jonathan Terhorst

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.