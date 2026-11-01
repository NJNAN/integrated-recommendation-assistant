# -*- coding: utf-8 -*-
"""v125：在修复版（干净）底座上重扫融合权重。

为什么需要
----------
旧权重 .15/.42/.215/.215（头部）是 103 次线上提交在**旧模型**上调出来的。
修复版把 H94/H95 换成了在干净特征上重训的版本，且新双版本更解耦
（ρ 0.9598 vs 旧 0.979）——最优权重必然移动。本次线上 0.8027 是在
"新模型 + 旧权重"下的成绩，不能代表干净底座的上限。

扫描设计（每一步只动一个方向，避免互相掩盖）
--------------------------------------------
A 组 头部权重：把权重从 binary(旧模型) 移向新的 H94/H95 双版本
B 组 尾部权重：池权重与 H94 的再平衡
C 组 对照组：复现当前线上配方

所有文件写到 outputs/competition_v125_reweight_clean/，逐个投即可，
一旦某方向不再提升就停。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/competition_v65_capacity/cache"
IN = ROOT / "outputs/competition_v120_H2_fixed"
V124 = ROOT / "outputs/competition_v124_headfocus_fixed"
OUT = ROOT / "outputs/competition_v125_reweight_clean"
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

CUR_HEAD = {"ours": .15, "binary": .42, "H94": .215, "H95": .215}
CUR_TAIL = {"ours": .15, "binary": .10, "H94": .28, "H95": .10,
            "H2L15": .07, "H2L7": .07, "p07s07": .07, "p11s24": .07, "p15s24": .07}

# A 组：头部权重移动（尾部保持当前配方）
A_GROUP = {
    "A1_ours15_bin36_2424": {"ours": .15, "binary": .36, "H94": .245, "H95": .245},
    "A2_ours15_bin30_2727": {"ours": .15, "binary": .30, "H94": .275, "H95": .275},
    "A3_ours10_bin42_2424": {"ours": .10, "binary": .42, "H94": .24,  "H95": .24},
    "A4_ours15_bin42_2815": {"ours": .15, "binary": .42, "H94": .28,  "H95": .15},
    "A5_ours05_bin42_2424": {"ours": .05, "binary": .42, "H94": .265, "H95": .265},
}
# B 组：尾部再平衡（头部保持当前配方）
B_GROUP = {
    "B1_tail_pool40_h94_23": {"ours": .15, "binary": .10, "H94": .23, "H95": .10,
                              "H2L15": .08, "H2L7": .08, "p07s07": .08, "p11s24": .08, "p15s24": .08},
    "B2_tail_pool30_h94_33": {"ours": .15, "binary": .10, "H94": .33, "H95": .10,
                              "H2L15": .064, "H2L7": .064, "p07s07": .064, "p11s24": .064, "p15s24": .064},
    "B3_tail_h95_20_pool25": {"ours": .15, "binary": .10, "H94": .25, "H95": .20,
                              "H2L15": .06, "H2L7": .06, "p07s07": .06, "p11s24": .06, "p15s24": .06},
}


def main() -> None:
    t0 = time.time()
    cand = read_candidates(ROOT / "data/sample_candidates.csv")
    uid = cand.user_id.to_numpy()
    k = (cand.user_id + "|" + cand.item_id).to_numpy()

    S = {}
    for n, p in PATHS.items():
        d = pd.read_csv(p, dtype={"user_id": str, "item_id": str})
        d["k"] = d.user_id + "|" + d.item_id
        x = d.set_index("k").score.reindex(cand.user_id + "|" + cand.item_id).to_numpy(float)
        s, g = pd.Series(x), pd.Series(uid)
        S[n] = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
    print(f"载入 {len(S)} 个来源", flush=True)

    def emit(tag, head, tail, cut=10):
        remaining = pd.DataFrame({"u": uid, "k": k})
        picked = []
        for i, wt in enumerate([head, tail]):
            sc = sum(w * S[n] for n, w in wt.items())
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
        sc = fin.set_index("k").score.reindex(cand.user_id + "|" + cand.item_id).to_numpy(float)
        sub = pd.DataFrame({"user_id": cand.user_id.to_numpy(),
                            "item_id": cand.item_id.to_numpy(), "score": sc})
        validate_submission(cand, sub)
        f = OUT / f"submission_v125_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:44s} sha={file_sha256(f)[:16]}", flush=True)

    print("=== C 组：对照组（复现当前线上配方 0.8027）===")
    emit("C0_repro_online", CUR_HEAD, CUR_TAIL)
    print("=== A 组：头部权重（把权重从 binary 移向新 H94/H95）===")
    for tag, h in A_GROUP.items():
        emit(tag, h, CUR_TAIL)
    print("=== B 组：尾部再平衡 ===")
    for tag, t in B_GROUP.items():
        emit(tag, CUR_HEAD, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
