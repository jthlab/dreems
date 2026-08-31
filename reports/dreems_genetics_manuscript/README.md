# DREEMS manuscript

This directory contains the manuscript, the aggregate evidence snapshots used
in the text, and the rendered figures and tables. The frozen evidence was
generated with the predecessor checkpoint whose SHA-256 is:

```text
f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4
```

That hash is intentionally retained by the manuscript checks. The packaged
production checkpoint now includes the surface gate and has different
parameters. Do not attribute these frozen numerical results to the newer model
without rebuilding the source analyses.

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

All generated manuscript figures are stored as PDF. Empirical comparison
panels may contain embedded raster source images when no vector original was
available, but the manuscript asset and LaTeX input remain PDF.

The source H2H implementation is `../../workflow/h2h.Snakefile`. The source
training and calibration workflows are `../../workflow/Snakefile` and
`../../workflow/calibration.Snakefile`.
