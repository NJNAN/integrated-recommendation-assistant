# -*- coding: utf-8 -*-
"""v126：按 A 组的反向趋势，加大旧来源权重 / 减少 ours，并配一组头部+尾部联合。

依据（线上实测）
----------------
C0 (ours.15 binary.42 H94.215 H95.215) = 0.8024
A1 (ours.15 binary.36 H94.245 H95.245) = 0.8020   ↓
A2 (ours.15 binary.30 H94.275 H95.275) = 0.8014   ↓↓
=> 单调：把权重从 binary 移向新 H94/H95 会持续变差，且 NDCG 同步跌。
   故本轮反向加大 binary，并同时测减小 ours 的方向（A3/A5 尚未投）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs/competition_v120_H2_fixed"
V124 = ROOT / "outputs/competition_v124_headfocus_fixed"
OUT = ROOT / "outputs/competition_v126_head_push"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

PATHS = {
    "ours":   ROOT / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": ROOT / "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    "H94":    IN / "submission_v120_H94fixed.csv",
    "H95":    IN / "submission_v120_H95fixed.csv",
    "H2L15":  ROOT / "outputs/competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "H2L7":   ROOT / "outputs/competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "p07s07": V124 / "submission_v124_H2L07s07.csv",
    "p11s24": V124 / "submission_v124_H2L11s24.csv",
    "p15s24": V124 / "submission_v124_H2L15s24.csv",
}
CUR_TAIL = {"ours": .15, "binary": .10, "H94": .28, "H95": .10,
            "H2L15": .07, "H2L7": .07, "p07s07": .07, "p11s24": .07, "p15s24": .07}
# 尾部：加大新模型、减小旧来源
TAIL_NEWHEAVY = {"ours": .10, "binary": .10, "H94": .24, "H95": .10,
                 "H2L15": .09, "H2L7": .09, "p07s07": .09, "p11s24": .09, "p15s24": .09}
TAIL_OURS0 = {"ours": .00, "binary": .10, "H94": .36, "H95": .14,
              "H2L15": .08, "H2L7": .08, "p07s07": .08, "p11s24": .08, "p15s24": .08}

JOBS = [
    # D 组：反向加大 binary（头部）
    ("D1_head_binary48", {"ours": .15, "binary": .48, "H94": .185, "H95": .185}, CUR_TAIL),
    ("D2_head_binary52", {"ours": .15, "binary": .52, "H94": .165, "H95": .165}, CUR_TAIL),
    ("D3_head_ours20",   {"ours": .20, "binary": .42, "H94": .19,  "H95": .19},  CUR_TAIL),
    # E 组：尾部换向（新模型加重 / ours 清零）
    ("E1_tail_newheavy", {"ours": .15, "binary": .42, "H94": .215, "H95": .215}, TAIL_NEWHEAVY),
    ("E2_tail_ours0",    {"ours": .15, "binary": .42, "H94": .215, "H95": .215}, TAIL_OURS0),
]


def main() -> None:
    t0 = time.time()
    cand = read_candidates(ROOT / "data/sample_candidates.csv")
    uid = cand.user_id.to_numpy()
    key = cand.user_id + "|" + cand.item_id
    k = key.to_numpy()

    S = {}
    for n, p in PATHS.items():
        d = pd.read_csv(p, dtype={"user_id": str, "item_id": str})
        d["k"] = d.user_id + "|" + d.item_id
        x = d.set_index("k").score.reindex(key).to_numpy(float)
        s, g = pd.Series(x), pd.Series(uid)
        S[n] = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()

    def emit(tag, head, tail, cut=10):
        remaining = pd.DataFrame({"u": uid, "k": k})
        picked = []
        for i, wt in enumerate([head, tail]):
            sc = sum(w * S[n] for n, w in wt.items() if w)
            remaining["sc"] = sc[remaining.index.to_numpy()]
            ordered = remaining.sort_values(["u", "sc"], ascending=[True, False], kind="mergesort")
            part = (ordered if i == 1 else ordered.groupby("u", sort=False).head(cut)).copy()
            part["_piece"] = i
            picked.append(part.drop(columns="sc"))
            remaining = remaining.drop(index=part.index)
        assert len(remaining) == 0
        fin = (pd.concat(picked, ignore_index=True)
               .sort_values(["u", "_piece"], kind="mergesort").reset_index(drop=True))
        fin["score"] = -fin.groupby("u").cumcount()
        sc = fin.set_index("k").score.reindex(key).to_numpy(float)
        sub = pd.DataFrame({"user_id": cand.user_id.to_numpy(),
                            "item_id": cand.item_id.to_numpy(), "score": sc})
        validate_submission(cand, sub)
        f = OUT / f"submission_v126_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:42s} sha={file_sha256(f)[:16]}", flush=True)

    print("=== D 组：反向加大旧来源（binary）===")
    for tag, h, t in JOBS[:3]:
        emit(tag, h, t)
    print("=== E 组：尾部换向 ===")
    for tag, h, t in JOBS[3:]:
        emit(tag, h, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
