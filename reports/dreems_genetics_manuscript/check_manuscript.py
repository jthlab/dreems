from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
EXPECTED_CHECKPOINT_SHA256 = "f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4"
EXPECTED_FEEMS_PROTOCOL = "v2_joint_lambda_lambda_q_leave_one_observed_deme_out"

REQUIRED_TABLES = (
    "model_setup.tex",
    "training_corpus.tex",
    "h2h_overall.tex",
    "h2h_paired.tex",
    "h2h_regimes.tex",
    "h2h_graph_sizes.tex",
    "runtime_overall.tex",
    "runtime_atomized.tex",
    "calibration.tex",
    "uncertainty_utility.tex",
    "empirical_datasets.tex",
    "wolf_empirical.tex",
)
REQUIRED_FIGURES = (
    "method_overview.pdf",
    "training_curve.pdf",
    "h2h_scatter_corr_mse.pdf",
    "h2h_regime_delta_bars.pdf",
    "h2h_examples_random.png",
    "h2h_examples_worst_mse.png",
    "runtime_scaling.pdf",
    "calibration_utility.pdf",
    "empirical_dreems_gallery.pdf",
    "empirical_published_comparison.pdf",
    "wolf_comparison.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{relative} must contain a JSON object")
    return value


def close(value: Any, expected: float, *, tolerance: float = 5e-4) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def check_required_files(errors: list[str]) -> None:
    required = (
        "main.tex",
        "methods.tex",
        "main.pdf",
        "main.log",
        "references.bib",
        "build_assets.py",
        "check_manuscript.py",
        "asset_summary.json",
        "analysis_summary.json",
        "release_manifest.json",
        "data/h2h_summary.csv",
        "data/h2h_protocol.json",
        "data/h2h_release_summary.json",
        "data/h2h_gallery_selection.json",
        "data/training_summary.json",
        "data/training_history.jsonl",
        "data/training_benchmark.tsv",
        "data/training_corpus_summary.json",
        "data/calibration_verification.json",
        "data/real_world_manifest.json",
        "data/wolf_dreems_current.json",
        "data/wolf_feems_summary.json",
        "data/wolf_current_summary.json",
        "data/runtime_examples.csv",
        "data/calibration_examples.csv",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for name in REQUIRED_TABLES:
        if not (ROOT / "tables" / name).is_file():
            errors.append(f"missing required table: {name}")
    for name in REQUIRED_FIGURES:
        if not (ROOT / "figures" / name).is_file():
            errors.append(f"missing required figure: {name}")


def check_manifest(errors: list[str]) -> None:
    path = ROOT / "release_manifest.json"
    if not path.is_file():
        return
    manifest = load_json("release_manifest.json")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        errors.append("release manifest has the wrong checkpoint hash")
    if manifest.get("feems_protocol") != EXPECTED_FEEMS_PROTOCOL:
        errors.append("release manifest has the wrong FEEMS protocol")
    if int(manifest.get("h2h_examples", -1)) != 512:
        errors.append("release manifest does not report 512 H2H examples")
    if int(manifest.get("empirical_datasets", -1)) != 15:
        errors.append("release manifest does not report 15 empirical datasets")
    records = manifest.get("records", [])
    if not isinstance(records, list) or not records:
        errors.append("release manifest has no artifact records")
        return
    seen: set[Path] = set()
    for record in records:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute():
            errors.append(f"manifest path is not repository-relative: {relative}")
            continue
        artifact = (REPO_ROOT / relative).resolve()
        try:
            artifact.relative_to(REPO_ROOT)
        except ValueError:
            errors.append(f"manifest path escapes the repository: {relative}")
            continue
        expected = record.get("sha256")
        if artifact in seen:
            errors.append(f"duplicate manifest artifact: {artifact}")
        seen.add(artifact)
        if not artifact.is_file():
            errors.append(f"manifest artifact is missing: {artifact}")
        elif expected != sha256(artifact):
            errors.append(f"manifest hash mismatch: {artifact}")

    derived_inputs = (
        "data/runtime_examples.csv",
        "data/resource_accounting.json",
        "data/calibration_examples.csv",
        "data/calibration_summary.json",
        "data/wolf_current_summary.json",
    )
    for relative in derived_inputs:
        if (ROOT / relative).resolve() not in seen:
            errors.append(f"derived manuscript input is absent from manifest: {relative}")


def check_evidence(errors: list[str]) -> None:
    if not (ROOT / "asset_summary.json").is_file():
        return
    assets = load_json("asset_summary.json")
    if assets.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        errors.append("asset summary has the wrong checkpoint hash")
    if int(assets.get("empirical_dataset_count", -1)) != 15:
        errors.append("empirical dataset count is not 15")
    if int(assets.get("h2h", {}).get("examples", -1)) != 512:
        errors.append("head-to-head example count is not 512")
    if assets.get("h2h", {}).get("feems_protocol") != EXPECTED_FEEMS_PROTOCOL:
        errors.append("asset summary has the wrong FEEMS protocol")
    if int(assets.get("calibration", {}).get("verification_examples", -1)) != 1024:
        errors.append("calibration verification count is not 1,024")

    analysis = load_json("analysis_summary.json")
    if analysis.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        errors.append("analysis summary has the wrong checkpoint hash")
    training = analysis.get("training", {})
    for field, expected in (("examples", 61_440), ("best_step", 29_000), ("final_step", 41_000)):
        if int(training.get(field, -1)) != expected:
            errors.append(f"training {field} does not equal {expected}")
    if not close(training.get("best_validation_loss"), 0.1572438609728124, tolerance=1e-10):
        errors.append("best validation loss is inconsistent")

    h2h = analysis.get("h2h", {})
    methods = h2h.get("methods", {})
    expected_metrics = {
        ("neural", "corr"): 0.697558,
        ("neural", "mse"): 0.557616,
        ("refined", "corr"): 0.719914,
        ("refined", "mse"): 0.533348,
        ("feems", "corr"): 0.614075,
        ("feems", "mse"): 0.738174,
    }
    for (method, metric), expected in expected_metrics.items():
        if not close(methods.get(method, {}).get(metric), expected):
            errors.append(f"unexpected H2H {method} {metric}")
    comparison = h2h.get("comparisons", {}).get("refined_vs_feems_corr", {})
    if int(comparison.get("wins", -1)) != 419:
        errors.append("refined-vs-FEEMS correlation wins do not equal 419")
    comparison = h2h.get("comparisons", {}).get("refined_vs_feems_mse", {})
    if int(comparison.get("wins", -1)) != 429:
        errors.append("refined-vs-FEEMS MSE wins do not equal 429")

    calibration = analysis.get("calibration", {}).get("verification", {})
    if int(calibration.get("verification_examples", -1)) != 1024:
        errors.append("analysis calibration verification count is not 1,024")
    selected_all = next(
        (
            row
            for row in calibration.get("rows", [])
            if row.get("method") == "selected" and row.get("sampling") == "all"
        ),
        None,
    )
    if selected_all is None or not close(selected_all.get("graph_balanced_coverage_95"), 0.957355):
        errors.append("verification 95% coverage is inconsistent")

    real = load_json("data/real_world_manifest.json")
    if len(real.get("datasets", [])) != 15:
        errors.append("real-world manifest does not contain 15 datasets")

    protocol = load_json("data/h2h_protocol.json")
    if protocol.get("dreems", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        errors.append("H2H protocol has the wrong checkpoint hash")
    if protocol.get("feems", {}).get("protocol") != EXPECTED_FEEMS_PROTOCOL:
        errors.append("H2H protocol is not documented joint LOO-CV")
    if protocol.get("feems", {}).get("cv_folds") != "one per observed deme":
        errors.append("H2H protocol does not hold out one observed deme per fold")

    h2h_path = ROOT / "data/h2h_summary.csv"
    if h2h_path.is_file():
        with h2h_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 512:
            errors.append(f"head-to-head CSV has {len(rows)} rows, expected 512")
        for index, row in enumerate(rows):
            if row.get("feems_protocol") != EXPECTED_FEEMS_PROTOCOL:
                errors.append(f"FEEMS row {index} has the wrong protocol")
                break
            for field in (
                "dreems_map_edge_log_corr",
                "dreems_map_edge_log_mse",
                "feems_edge_log_corr",
                "feems_edge_log_mse",
            ):
                try:
                    finite = math.isfinite(float(row[field]))
                except (KeyError, TypeError, ValueError):
                    finite = False
                if not finite:
                    errors.append(f"H2H row {index} has invalid {field}")
                    break


def check_manuscript_text(errors: list[str]) -> None:
    path = ROOT / "main.tex"
    if not path.is_file():
        return
    main_text = path.read_text()
    methods_path = ROOT / "methods.tex"
    methods_text = methods_path.read_text() if methods_path.is_file() else ""
    text = main_text + "\n" + methods_text
    normalized = re.sub(r"\s+", " ", text)
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        main_text,
        flags=re.DOTALL,
    )
    if abstract_match is None:
        errors.append("manuscript has no abstract")
    else:
        abstract = re.sub(r"\\(?:dreems|feems)", "DREEMS", abstract_match.group(1))
        abstract = abstract.replace(r"\%", "%")
        words = re.findall(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", abstract)
        if len(words) > 250:
            errors.append(f"abstract has {len(words)} words, exceeding the 250-word limit")

    required_sections = (
        r"\begin{abstract}",
        r"\section{Introduction}",
        r"\section{Results}",
        r"\section{Discussion}",
        r"\section{Materials and methods}",
        r"\subsection{Model description}",
        r"\subsection{All-pairs linearized resistance initializer}",
        r"\subsection{Fixed diffusion-band features}",
        r"\subsection{Graph-balanced training objective}",
        r"\subsection{Fixed q-aware trust-region covariance refinement}",
        r"\subsection{Synthetic head-to-head benchmark}",
        r"\subsection{Predictive-SD calibration and utility}",
        r"\subsection{Empirical reanalyses}",
    )
    for section in required_sections:
        if section not in text:
            errors.append(f"missing manuscript section: {section}")

    ordered_sections = (
        r"\section{Introduction}",
        r"\section{Results}",
        r"\section{Discussion}",
        r"\section{Materials and methods}",
    )
    positions = [main_text.find(section) for section in ordered_sections]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append(
            "manuscript sections do not follow Introduction, Results, Discussion, Methods"
        )

    required_phrases = (
        "documented leave-one-observed-deme-out",
        "Every component is reduced within each graph",
        "not the result of variational inference",
        "no Gaussian-prior penalty",
        "true conductance fields are unknown",
        "Parallelism was not matched and favors",
        "one-GPU run",
        "15 public datasets",
        "4,096 held-out examples",
        EXPECTED_CHECKPOINT_SHA256,
        "aa890b0e8ef583dd657a6a696af591b732d1d8be",
    )
    for phrase in required_phrases:
        if phrase not in normalized:
            errors.append(f"missing required clarification: {phrase}")

    stale_claims = (
        "15b987a889f785ebdb81213677114ce5b561ae7d852677f1645ccecc1034011b",
        "second, independent genotype replicate",
        "six public datasets",
        "mean edge correlation 0.654",
        "mean squared error 0.665",
        "14.9 s for",
        "585.5 s",
        "one NVIDIA A40 GPU",
        "step 53,000",
        "step 65,000",
        "SNP-CV",
        "unbounded-equivalent",
    )
    for claim in stale_claims:
        if claim in normalized:
            errors.append(f"stale manuscript claim remains: {claim}")
    for pending in ("TODO", "FIXME", "placeholder"):
        if pending in text:
            errors.append(f"pending text remains: {pending}")


def check_latex_log(errors: list[str]) -> None:
    path = ROOT / "main.log"
    if not path.is_file():
        errors.append("missing main.log")
        return
    issue_patterns = (
        re.compile(r"^!"),
        re.compile(r"LaTeX Warning: (Citation|Reference).*undefined"),
        re.compile(r"There were undefined references"),
        re.compile(r"Overfull \\hbox"),
        re.compile(r"Overfull \\vbox"),
    )
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if any(pattern.search(line) for pattern in issue_patterns):
            errors.append(f"LaTeX log issue at line {line_number}: {line}")


def main() -> None:
    errors: list[str] = []
    check_required_files(errors)
    check_manifest(errors)
    check_evidence(errors)
    check_manuscript_text(errors)
    check_latex_log(errors)
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, indent=2))
        sys.exit(1)
    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
