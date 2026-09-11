# DREEMS Island Simulator Extension

This is the optional Rust/PyO3 backend for `dreems.data.island_sim`.

Build it into the active Python environment with:

```bash
module load rust/1.92.0
uv sync --group rust --group dev
uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml
```

The Python package works without this extension by using a reference
implementation, but full corpus generation should use the Rust backend.
If you rerun `uv sync`, rerun `maturin develop` afterward because sync may
remove the editable extension from the environment.
