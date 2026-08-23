# DREEMS manuscript

This directory contains the manuscript, the aggregate evidence snapshots used
in the text, and the rendered figures and tables. The released checkpoint is
`../../src/dreems/data/dreems_diffusion_bands.dreems.zip`, with SHA-256:

```text
f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4
```

## Build and check

From the repository root:

```bash
uv sync --frozen --group manuscript
make -C reports/dreems_genetics_manuscript check
```

The ordinary `pdf` and `check` targets use the committed result snapshots and
figures. They do not rerun training or FEEMS.

## Refresh source analyses

`make pdf-from-sources` first refreshes manuscript assets from the full
training, H2H, calibration, and empirical result directories. Those artifacts
are not stored in Git because they are large. `build_assets.py` documents the
default Great Lakes and Turbo locations. Set its `DREEMS_*` environment
variables when the archived results live elsewhere.

The frozen publication evidence includes:

- `data/h2h_summary.csv`: all 512 paired DREEMS and FEEMS results;
- `data/h2h_protocol.json`: checkpoint, refinement, and FEEMS CV provenance;
- `data/training_summary.json` and `data/training_history.jsonl`: the selected
  checkpoint and complete logged optimization trajectory;
- `data/calibration_verification.json`: untouched posterior-SD verification;
- `data/real_world_manifest.json`: the empirical reanalysis inventory;
- `analysis_summary.json`: the numerical claims consumed by the manuscript;
- `release_manifest.json`: hashes for the compiled evidence bundle.

The source H2H implementation is `../../workflow/h2h.Snakefile`. The source
training and calibration workflows are `../../workflow/Snakefile` and
`../../workflow/calibration.Snakefile`.
