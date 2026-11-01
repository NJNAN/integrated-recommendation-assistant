"""Reproduce the v2 final assembly (R1_head48, online composite 0.8060).

Version 2 is the audited rebuild.  The v1 line (reproduce_oblivionis.py) locks the
pre-audit artifact 0.8049; this one locks the post-fix artifact 0.8060.

What v2 changes relative to v1
------------------------------
1. Engine temporal crossover fixed: each historical window now rebuilds its own
   engine from that window's history (see scripts/experiment_v120_H2_fixed.py in
   source/code/), so the sources H2f_r120_s42 / H2f_r300_s42 are trained on
   window-consistent features.
2. Row misalignment fixed: feature frames keep every candidate row and are merged
   by (window, user_id, item_id) instead of by DataFrame row number
   (see scripts/fixed_alignment.py).
3. Seven candidate-statistic features restored (candidate_frame passed).

The nine frozen sources and the two-segment recipe below are exactly what the
online submission used.

Usage:
    python reproduce_v2.py
    python verify_oblivionis_package.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "Oblivionis"
IMPLEMENTATION = "R1_head48"
EXPECTED_SHA256 = "d924f90e7b3c661614738ecc6353882f6497508338fc85153b2dd555a4cd66a0"
EXPECTED_ROWS = 58_205
EXPECTED_USERS = 736

COMPONENT_FILES = {
    "ours":   "ours.csv",
    "binary": "binary.csv",
    "R120":   "H2f_r120_s42.csv",
    "R300":   "H2f_r300_s42.csv",
    "nL15":   "v121_H2L15.csv",
    "nL7":    "v122_H2L7.csv",
    "n07":    "v124_H2L07s07.csv",
    "n11":    "v124_H2L11s24.csv",
    "n15":    "v124_H2L15s24.csv",
}

PROJECT_COMPONENTS = {
    "ours":   "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    "R120":   "outputs/competition_v120_H2_fixed/submission_v120_H2f_r120_s42.csv",
    "R300":   "outputs/competition_v120_H2_fixed/submission_v120_H2f_r300_s42.csv",
    "nL15":   "outputs/competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "nL7":    "outputs/competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "n07":    "outputs/competition_v124_headfocus_fixed/submission_v124_H2L07s07.csv",
    "n11":    "outputs/competition_v124_headfocus_fixed/submission_v124_H2L11s24.csv",
    "n15":    "outputs/competition_v124_headfocus_fixed/submission_v124_H2L15s24.csv",
}

HEAD_WEIGHTS = {"ours": .10, "binary": .23, "R120": .48, "R300": .19}
TAIL_WEIGHTS = {"ours": .10, "binary": .08, "R120": .30, "R300": .10,
                "nL15": .084, "nL7": .084, "n07": .084, "n11": .084, "n15": .084}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_layout(script_path: Path):
    root = script_path.parent
    pkg = root / "inputs" / "components_v2"
    if pkg.is_dir():
        return (root / "inputs" / "sample_candidates.csv",
                root / "output" / "submission_v2_R1_head48.csv",
                {n: pkg / f for n, f in COMPONENT_FILES.items()},
                root / "expected" / "submission_v2_R1_head48.csv")
    proj = root.parents[1]
    return (proj / "data" / "sample_candidates.csv",
            proj / "outputs" / "competition_v131_refine" / "submission_v131_R1_head48_reproduced.csv",
            {n: proj / r for n, r in PROJECT_COMPONENTS.items()},
            proj / "outputs" / "competition_v131_refine" / "submission_v131_R1_head48.csv")


def user_percentile(values: np.ndarray, users: np.ndarray) -> np.ndarray:
    s, g = pd.Series(values), pd.Series(users)
    return (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_cand, default_out, defaults, default_ref = detect_layout(Path(__file__).resolve())
    parser.add_argument("--candidates", type=Path, default=default_cand)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--reference", type=Path, default=default_ref)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates, dtype={"user_id": str, "item_id": str})[["user_id", "item_id"]]
    cand["k"] = cand.user_id + "|" + cand.item_id
    if cand["k"].duplicated().any():
        raise ValueError("candidate file contains duplicate user-item pairs")
    uid, key = cand.user_id.to_numpy(), cand.k.to_numpy()

    comps = defaults
    if args.components_dir:
        comps = {n: args.components_dir / f for n, f in COMPONENT_FILES.items()}

    S = {}
    for n, p in comps.items():
        if not p.is_file():
            raise FileNotFoundError(f"missing component {n}: {p}")
        d = pd.read_csv(p, dtype={"user_id": str, "item_id": str})
        d["k"] = d.user_id + "|" + d.item_id
        x = d.set_index("k").score.reindex(cand.k).to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"component {n} has missing scores")
        S[n] = user_percentile(x, uid)

    remaining = pd.DataFrame({"u": uid, "k": key})
    picked = []
    for i, wt in enumerate([HEAD_WEIGHTS, TAIL_WEIGHTS]):
        sc = sum(w * S[n] for n, w in wt.items() if w)
        remaining["sc"] = sc[remaining.index.to_numpy()]
        ordered = remaining.sort_values(["u", "sc"], ascending=[True, False], kind="mergesort")
        part = (ordered if i == 1 else ordered.groupby("u", sort=False).head(10)).copy()
        part["_piece"] = i
        picked.append(part.drop(columns="sc"))
        remaining = remaining.drop(index=part.index)
    if not remaining.empty:
        raise AssertionError("not every candidate was assigned to a segment")
    fin = (pd.concat(picked, ignore_index=True)
           .sort_values(["u", "_piece"], kind="mergesort").reset_index(drop=True))
    fin["score"] = -fin.groupby("u").cumcount()
    score = fin.set_index("k").score.reindex(cand.k).to_numpy(float)
    out = pd.DataFrame({"user_id": cand.user_id.to_numpy(),
                        "item_id": cand.item_id.to_numpy(), "score": score})

    if len(out) != EXPECTED_ROWS or out.user_id.nunique() != EXPECTED_USERS:
        raise ValueError(f"unexpected shape: rows={len(out)} users={out.user_id.nunique()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.6f")

    digest = sha256(args.output)
    ref = {"reference": str(args.reference), "exists": args.reference.is_file()}
    if args.reference.is_file():
        e = pd.read_csv(args.reference, dtype={"user_id": str, "item_id": str})
        p = pd.read_csv(args.output, dtype={"user_id": str, "item_id": str})
        ref.update({"same_pairs": p[["user_id", "item_id"]].equals(e[["user_id", "item_id"]]),
                    "max_score_delta": float(np.max(np.abs(p.score.to_numpy(float) - e.score.to_numpy(float)))),
                    "exact_bytes": args.output.read_bytes() == args.reference.read_bytes()})
    report = {"model": f"{MODEL_NAME}({IMPLEMENTATION})", "rows": len(out),
              "users": int(out.user_id.nunique()), "output": str(args.output),
              "sha256": digest, "expected_sha256": EXPECTED_SHA256,
              "expected_hash_match": digest == EXPECTED_SHA256,
              "reference_check": ref,
              "head_weights": HEAD_WEIGHTS, "tail_weights": TAIL_WEIGHTS}
    rp = args.report or args.output.with_suffix(".verification.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if digest != EXPECTED_SHA256:
        raise SystemExit("ERROR: output hash does not match the locked v2 submission")
    if ref.get("exists") and not ref.get("exact_bytes"):
        raise SystemExit("ERROR: output is not byte-identical to the reference")


if __name__ == "__main__":
    main()
