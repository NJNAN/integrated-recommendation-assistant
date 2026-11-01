# -*- coding: utf-8 -*-
"""v131：围绕 Q2（线上 0.8059，新纪录）做最后精细搜索。

线上实测
--------
Q1  R120头.215                 0.8049
Q2  R120头.45                  0.8059  ← 峰值（NDCG 0.8991 破纪录）
Q3  R120头.55                  0.8056
Q4  掺旧模型（头.18）           0.8050  → 干净版 > 混合版

两条待榨的方向：
  (a) 头部：R120 权重峰值在 .45~.55 之间，试 .48 / .50
  (b) 尾部：Q4 的 GAUC 0.8796 > Q2 的 0.8792，说明尾部结构还能调
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
O = ROOT / "outputs"
OUT = ROOT / "outputs/competition_v131_refine"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

PATHS = {
    "ours":   O / "competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": O / "competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    "R120":   IN / "submission_v120_H2f_r120_s42.csv",
    "R300":   IN / "submission_v120_H2f_r300_s42.csv",
    "R300s7": IN / "submission_v120_H2f_r300_s07.csv",
    "nL15":   O / "competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "nL7":    O / "competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "n07":    V124 / "submission_v124_H2L07s07.csv",
    "n11":    V124 / "submission_v124_H2L11s24.csv",
    "n15":    V124 / "submission_v124_H2L15s24.csv",
}
Q2_H = {"ours": .10, "binary": .25, "R120": .45, "R300": .20}
Q2_T = {"ours": .10, "binary": .08, "R120": .30, "R300": .10,
        "nL15": .084, "nL7": .084, "n07": .084, "n11": .084, "n15": .084}


def renorm(w, total=1.0):
    s = sum(w.values())
    return {k: v * total / s for k, v in w.items() if v}


JOBS = [
    # R1/R2：头部峰位精调
    ("R1_head48", {"ours": .10, "binary": .23, "R120": .48, "R300": .19}, Q2_T),
    ("R2_head50", {"ours": .10, "binary": .22, "R120": .50, "R300": .18}, Q2_T),
    # R3：尾部加重池（Q4 的 GAUC 更高，池可能是原因）
    ("R3_tail_pool095", Q2_H,
     renorm({"ours": .07, "binary": .05, "R120": .30, "R300": .10,
             "nL15": .095, "nL7": .095, "n07": .095, "n11": .095, "n15": .095})),
    # R4：尾部加重 R120（它是全线最强来源）
    ("R4_tail_R120", Q2_H,
     renorm({"ours": .05, "binary": .05, "R120": .40, "R300": .12,
             "nL15": .076, "nL7": .076, "n07": .076, "n11": .076, "n15": .076})),
    # R5：头部峰位 + 尾部加重 R120 的合并
    ("R5_combined", {"ours": .10, "binary": .23, "R120": .48, "R300": .19},
     renorm({"ours": .05, "binary": .05, "R120": .40, "R300": .12,
             "nL15": .076, "nL7": .076, "n07": .076, "n11": .076, "n15": .076})),
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
        f = OUT / f"submission_v131_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:40s} sha={file_sha256(f)[:16]}", flush=True)

    for tag, h, t in JOBS:
        emit(tag, h, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
