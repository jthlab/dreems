# Reproducing manuscript figures

To reproduce all the figures relating to wolves, Allohistium fishes, and blacklegged ticks, please see the documentation page and tutorials for DREEMS. All data sources and preprocessing procedures are documented in the tutorials.

For the yellow birch dataset, please visit the following website https://datadryad.org/dataset/doi:10.5061/dryad.wwpzgmsxc for the VCF, and all sampling locations are found in "Supplementary Material 2" document from the paper. We followed the author's recommended data preprocessing procedure, please run the tree_hexploid_preprocessing.R file with the downloaded VCF to obtain the filtered_tree_genotypes.vcf.

To reproduce all other figures, please see feems_snakefile and manuscript_snakefile. The snakemake commands will reproduce all figures. For the feems_snakefile, you must follow the installation procedure from Novembre lab FEEMS' github. For FEEMS instalation, first install the SuiteSparse/CHOLMOD development library required by `scikit-sparse`, then install the optional FEEMS dependencies. For example, on Ubuntu or Debian:

```bash
sudo apt-get install libsuitesparse-dev
uv sync --frozen --extra feems
```

On macOS with Homebrew, install SuiteSparse with `brew install suite-sparse`
before running the same `uv sync` command. The neural DREEMS estimator does not
require FEEMS or SuiteSparse.