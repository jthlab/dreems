# Project Instructions

## Workflow Orchestration

- Use Snakemake for pipeline execution, parallelism, caching, and artifact
  management. Do not submit one-off worker jobs directly when a Snakemake rule
  can represent the unit of work.
- Keep expensive transient artifacts on scratch storage. Keep only
  publication-ready source code, configuration, reports, and durable outputs in
  the project repository or turbo project directory.
- On Great Lakes, use the Slurm account `jonth0` by default. `jonth1` is a paid
  account and should be used only when explicitly requested.
- The local high-memory/high-core workstation target is `maturin`; use the local
  Snakemake profile or `workflow/run_maturin.sh` there.

## Manuscript

- After rebuilding the manuscript, copy the final PDF to
  `/home/jonth/turbo/dreems.pdf`.
