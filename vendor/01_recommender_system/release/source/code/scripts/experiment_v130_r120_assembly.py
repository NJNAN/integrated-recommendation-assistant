# -*- coding: utf-8 -*-
"""v130：用公平复刻模型（H2f_r120_s42）重建装配。

关键发现（线上实测，唯一公平对标点）
------------------------------------
    旧 H94（带病，120轮/s42）  0.8007   NDCG 0.8920  R 0.4699  GAUC 0.8747
    新 H2f_r120（干净，120轮/s42）0.8032   NDCG 0.8984  R 0.4706  GAUC 0.8744
                                  +0.0025     +0.0064    +0.0007    -0.0003

=> ① 干净化不亏，单模型反而 +0.0025（之前"亏 0.0021"是把 120 轮改成 300 轮造成的混淆）
   ② NDCG 0.8984 是全程最高（旧 DD2 系统 0.8962、CH4 0.8976）
   ③ 300 轮对 NDCG@10 有害，后续一律以 120 轮为准

因此本组把 H94/H95 槽换成 r120 系列，并围绕"最强头部来源"重新分配权重。
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
OUT = ROOT / "outputs/competition_v130_r120_assembly"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

PATHS = {
    "ours":   O / "competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": O / "competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    # 旧（带病）
    "oH94":   O / "competition_v94_O2_H2/submission_v94_H2.csv",
    "oH95":   O / "competition_v95_O2_H2_fixed/submission_v95_H2.csv",
    "oL15":   O / "competition_v102_H2_variants/submission_v102_H2L15.csv",
    "oL7":    O / "competition_v104_smallcap/submission_v104_H2L7.csv",
    "o07":    O / "competition_v106_headfocus/submission_v106_H2L07s07.csv",
    "o11":    O / "competition_v106_headfocus/submission_v106_H2L11s24.csv",
    "o15":    O / "competition_v106_headfocus/submission_v106_H2L15s24.csv",
    # 新（修复管线）—— R120 是主力
    "R120":   IN / "submission_v120_H2f_r120_s42.csv",
    "R300":   IN / "submission_v120_H2f_r300_s42.csv",
    "R300s7": IN / "submission_v120_H2f_r300_s07.csv",
    "nL15":   O / "competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "nL7":    O / "competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "n07":    V124 / "submission_v124_H2L07s07.csv",
    "n11":    V124 / "submission_v124_H2L11s24.csv",
    "n15":    V124 / "submission_v124_H2L15s24.csv",
}

# Q1：沿用旧 DD2 结构，只把 H94/H95 槽换成 r120 / r300
Q1_H = {"ours": .15, "binary": .42, "R120": .215, "R300": .215}
Q1_T = {"ours": .15, "binary": .10, "R120": .28, "R300": .10,
        "nL15": .07, "nL7": .07, "n07": .07, "n11": .07, "n15": .07}
# Q2：R120 加重（它是 NDCG 最强来源）
Q2_H = {"ours": .10, "binary": .25, "R120": .45, "R300": .20}
Q2_T = {"ours": .10, "binary": .08, "R120": .30, "R300": .10,
        "nL15": .084, "nL7": .084, "n07": .084, "n11": .084, "n15": .084}
# Q3：更极端 —— 头部几乎全给 R120 与 R300
Q3_H = {"ours": .08, "binary": .15, "R120": .55, "R300": .12, "R300s7": .10}
Q3_T = {"ours": .08, "binary": .08, "R120": .34, "R300": .10,
        "nL15": .08, "nL7": .08, "n07": .08, "n11": .08, "n15": .08}
# Q4：旧 DD2 + R120 作为额外来源（混合路线）
Q4_H = {"ours": .13, "binary": .35, "oH94": .17, "oH95": .17, "R120": .18}
Q4_T = {"ours": .13, "binary": .09, "oH94": .22, "oH95": .09,
        "oL15": .06, "oL7": .06, "o07": .06, "o11": .06, "o15": .06, "R120": .11}
# Q5：剔除 ours（去掉未修的 v36 分量）
Q5_H = {"binary": .45, "R120": .55}
Q5_T = {"binary": .15, "R120": .45,
        "nL15": .08, "nL7": .08, "n07": .08, "n11": .08, "n15": .08}

JOBS = [("Q1_r120pair", Q1_H, Q1_T), ("Q2_R120heavy", Q2_H, Q2_T),
        ("Q3_R120max", Q3_H, Q3_T), ("Q4_mix_old_r120", Q4_H, Q4_T),
        ("Q5_no_ours", Q5_H, Q5_T)]


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
    print(f"载入 {len(S)} 个来源", flush=True)

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
        f = OUT / f"submission_v130_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:42s} sha={file_sha256(f)[:16]}", flush=True)

    for tag, h, t in JOBS:
        emit(tag, h, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
