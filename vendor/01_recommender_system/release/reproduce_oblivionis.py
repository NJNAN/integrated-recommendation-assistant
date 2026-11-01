"""Reproduce the exact Oblivionis(dd2poolseed) final assembly.

This is the cleaned, single-file implementation of the assembly section from
experiment_v106_headfocus.py.  It can run inside the original project or from
the frozen reproduction package built by build_oblivionis_repro_package.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "Oblivionis"
IMPLEMENTATION = "dd2poolseed"
EXPECTED_SHA256 = "123816780ae366ee5e9bea43cd69d4188bc55fe8c12cd67c126b9923a25518e3"
EXPECTED_ROWS = 58_205
EXPECTED_USERS = 736

COMPONENT_FILES = {
    "ours": "ours.csv",
    "binary": "binary.csv",
    "H94": "H94.csv",
    "H95": "H95.csv",
    "H2L15": "H2L15.csv",
    "H2L7": "H2L7.csv",
    "H2L07s07": "H2L07s07.csv",
    "H2L11s24": "H2L11s24.csv",
    "H2L15s24": "H2L15s24.csv",
}

PROJECT_COMPONENTS = {
    "ours": "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    "H94": "outputs/competition_v94_O2_H2/submission_v94_H2.csv",
    "H95": "outputs/competition_v95_O2_H2_fixed/submission_v95_H2.csv",
    "H2L15": "outputs/competition_v102_H2_variants/submission_v102_H2L15.csv",
    "H2L7": "outputs/competition_v104_smallcap/submission_v104_H2L7.csv",
    "H2L07s07": "outputs/competition_v106_headfocus/submission_v106_H2L07s07.csv",
    "H2L11s24": "outputs/competition_v106_headfocus/submission_v106_H2L11s24.csv",
    "H2L15s24": "outputs/competition_v106_headfocus/submission_v106_H2L15s24.csv",
}

HEAD_WEIGHTS = {"ours": 0.15, "binary": 0.42, "H94": 0.215, "H95": 0.215}
TAIL_WEIGHTS = {
    "ours": 0.15,
    "binary": 0.10,
    "H94": 0.28,
    "H2L15": 0.07,
    "H2L7": 0.07,
    "H2L07s07": 0.07,
    "H2L11s24": 0.07,
    "H2L15s24": 0.07,
    "H95": 0.10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_layout(script_path: Path) -> tuple[Path, Path, dict[str, Path], Path]:
    """Return candidates, output, components, and optional reference defaults."""
    package_root = script_path.parent
    package_components = package_root / "inputs" / "components"
    if package_components.is_dir():
        return (
            package_root / "inputs" / "sample_candidates.csv",
            package_root / "output" / "submission_DD2_pool_seed.csv",
            {name: package_components / filename for name, filename in COMPONENT_FILES.items()},
            package_root / "expected" / "submission_DD2_pool_seed.csv",
        )

    project_root = script_path.parents[1]
    return (
        project_root / "data" / "sample_candidates.csv",
        project_root / "outputs" / "competition_v106_headfocus" / "submission_DD2_pool_seed_reproduced.csv",
        {name: project_root / rel for name, rel in PROJECT_COMPONENTS.items()},
        project_root / "outputs" / "competition_v106_headfocus" / "submission_DD2_pool_seed.csv",
    )


def read_candidates(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
    required = ["user_id", "item_id"]
    if not set(required).issubset(candidates.columns):
        raise ValueError(f"candidate file must contain {required}: {path}")
    candidates = candidates[required].copy()
    candidates["k"] = candidates["user_id"] + "|" + candidates["item_id"]
    if candidates["k"].duplicated().any():
        raise ValueError("candidate file contains duplicate user-item pairs")
    return candidates


def user_percentile(values: np.ndarray, users: np.ndarray) -> np.ndarray:
    score = pd.Series(values)
    group = pd.Series(users)
    return (
        score.groupby(group).rank(method="average")
        / score.groupby(group).transform("size")
    ).to_numpy()


def load_components(
    candidates: pd.DataFrame, component_paths: dict[str, Path]
) -> dict[str, np.ndarray]:
    candidate_keys = candidates["k"]
    users = candidates["user_id"].to_numpy()
    result: dict[str, np.ndarray] = {}
    for name, path in component_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing component {name}: {path}")
        frame = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
        if not {"user_id", "item_id", "score"}.issubset(frame.columns):
            raise ValueError(f"invalid component columns: {path}")
        frame["k"] = frame["user_id"] + "|" + frame["item_id"]
        if frame["k"].duplicated().any():
            raise ValueError(f"duplicate pairs in component {name}: {path}")
        aligned = frame.set_index("k")["score"].reindex(candidate_keys).to_numpy(float)
        if not np.isfinite(aligned).all():
            missing = int((~np.isfinite(aligned)).sum())
            raise ValueError(f"component {name} has {missing} missing/non-finite scores")
        result[name] = user_percentile(aligned, users)
    return result


def weighted_score(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weight * signals[name] for name, weight in weights.items())


def assemble(candidates: pd.DataFrame, signals: dict[str, np.ndarray], cutoff: int = 10) -> pd.DataFrame:
    users = candidates["user_id"].to_numpy()
    remaining = pd.DataFrame({"u": users, "k": candidates["k"].to_numpy()})
    pieces: list[pd.DataFrame] = []

    for piece_number, weights in enumerate((HEAD_WEIGHTS, TAIL_WEIGHTS)):
        score = weighted_score(signals, weights)
        remaining["sc"] = score[remaining.index.to_numpy()]
        ordered = remaining.sort_values(
            ["u", "sc"], ascending=[True, False], kind="mergesort"
        )
        part = (
            ordered.groupby("u", sort=False).head(cutoff)
            if piece_number == 0
            else ordered
        ).copy()
        part["_piece"] = piece_number
        pieces.append(part.drop(columns="sc"))
        remaining = remaining.drop(index=part.index)

    if not remaining.empty:
        raise AssertionError("not every candidate was assigned to a segment")

    ranked = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["u", "_piece"], kind="mergesort")
        .reset_index(drop=True)
    )
    ranked["score"] = -ranked.groupby("u").cumcount()
    score = ranked.set_index("k")["score"].reindex(candidates["k"]).to_numpy(float)
    return pd.DataFrame(
        {
            "user_id": candidates["user_id"].to_numpy(),
            "item_id": candidates["item_id"].to_numpy(),
            "score": score,
        }
    )


def validate_output(candidates: pd.DataFrame, submission: pd.DataFrame) -> dict[str, int]:
    if len(submission) != len(candidates):
        raise ValueError("submission row count differs from candidates")
    pairs = submission["user_id"].astype(str) + "|" + submission["item_id"].astype(str)
    if not pairs.equals(candidates["k"]):
        raise ValueError("submission pair order differs from candidates")
    if not np.isfinite(submission["score"].to_numpy(float)).all():
        raise ValueError("submission contains non-finite scores")
    users = int(submission["user_id"].nunique())
    if len(submission) != EXPECTED_ROWS or users != EXPECTED_USERS:
        raise ValueError(
            f"unexpected dataset shape: rows={len(submission)}, users={users}"
        )
    return {"rows": len(submission), "users": users}


def compare_reference(output: Path, reference: Path) -> dict[str, object]:
    result: dict[str, object] = {"reference": str(reference), "exists": reference.is_file()}
    if not reference.is_file():
        return result
    produced = pd.read_csv(output, dtype={"user_id": str, "item_id": str})
    expected = pd.read_csv(reference, dtype={"user_id": str, "item_id": str})
    same_pairs = produced[["user_id", "item_id"]].equals(expected[["user_id", "item_id"]])
    max_delta = (
        float(np.max(np.abs(produced["score"].to_numpy(float) - expected["score"].to_numpy(float))))
        if len(produced) == len(expected)
        else None
    )
    result.update(
        {
            "same_pairs": same_pairs,
            "max_score_delta": max_delta,
            "reference_sha256": sha256(reference),
            "exact_bytes": output.read_bytes() == reference.read_bytes(),
        }
    )
    return result


def main() -> None:
    default_candidates, default_output, defaults, default_reference = detect_layout(
        Path(__file__).resolve()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=default_candidates)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--reference", type=Path, default=default_reference)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    components = defaults
    if args.components_dir:
        components = {
            name: args.components_dir / filename
            for name, filename in COMPONENT_FILES.items()
        }

    candidates = read_candidates(args.candidates)
    signals = load_components(candidates, components)
    submission = assemble(candidates, signals)
    shape = validate_output(candidates, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, float_format="%.6f")

    digest = sha256(args.output)
    reference = compare_reference(args.output, args.reference)
    report = {
        "model": f"{MODEL_NAME}({IMPLEMENTATION})",
        **shape,
        "output": str(args.output),
        "sha256": digest,
        "expected_sha256": EXPECTED_SHA256,
        "expected_hash_match": digest == EXPECTED_SHA256,
        "reference_check": reference,
        "head_weights": HEAD_WEIGHTS,
        "tail_weights": TAIL_WEIGHTS,
    }
    report_path = args.report or args.output.with_suffix(".verification.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if digest != EXPECTED_SHA256:
        raise SystemExit("ERROR: output hash does not match the locked submission")
    if reference.get("exists") and not reference.get("exact_bytes"):
        raise SystemExit("ERROR: output is not byte-identical to the reference")


if __name__ == "__main__":
    main()
