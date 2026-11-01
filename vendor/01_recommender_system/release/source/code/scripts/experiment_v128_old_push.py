# -*- coding: utf-8 -*-
"""v128：围绕「旧 DD2 配方 + 新来源」做定点推进。

线上实测
--------
F1 全干净 (D2头 + E1尾)                  0.8028   NDCG .8934  R .4715  GAUC .8778
G1 旧H94/H95 并入(@.07头/.05尾)          0.8040   NDCG .8949  R .4719  GAUC .8791
旧 DD2（带病，榜上保留）                  0.8049   NDCG .8962  R .4736  GAUC .8791

读数：GAUC 已追平，差的全在 NDCG（头部）=> 继续加大旧来源的头部权重。
注意：G1 用的尾部池是**旧**池成员（v106 训练的），本组把新池也纳入比较。
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
OUT = ROOT / "outputs/competition_v128_old_push"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

OLD = ROOT / "outputs"
PATHS = {
    "ours":   OLD / "competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": OLD / "competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    # 旧（带病管线）
    "oH94":   OLD / "competition_v94_O2_H2/submission_v94_H2.csv",
    "oH95":   OLD / "competition_v95_O2_H2_fixed/submission_v95_H2.csv",
    "oL15":   OLD / "competition_v102_H2_variants/submission_v102_H2L15.csv",
    "oL7":    OLD / "competition_v104_smallcap/submission_v104_H2L7.csv",
    "o07s07": OLD / "competition_v106_headfocus/submission_v106_H2L07s07.csv",
    "o11s24": OLD / "competition_v106_headfocus/submission_v106_H2L11s24.csv",
    "o15s24": OLD / "competition_v106_headfocus/submission_v106_H2L15s24.csv",
    # 新（修复管线）
    "H94":    IN / "submission_v120_H94fixed.csv",
    "H95":    IN / "submission_v120_H95fixed.csv",
    "nL15":   OLD / "competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "nL7":    OLD / "competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "n07s07": V124 / "submission_v124_H2L07s07.csv",
    "n11s24": V124 / "submission_v124_H2L11s24.csv",
    "n15s24": V124 / "submission_v124_H2L15s24.csv",
}

OLD_HEAD = {"ours": .15, "binary": .42, "oH94": .215, "oH95": .215}
OLD_TAIL = {"ours": .15, "binary": .10, "oH94": .28, "oH95": .10,
            "oL15": .07, "oL7": .07, "o07s07": .07, "o11s24": .07, "o15s24": .07}


def renorm(w: dict, total: float = 1.0) -> dict:
    s = sum(w.values())
    return {k: v * total / s for k, v in w.items()}


JOBS = [
    # H0：对照 —— 完全复现旧 DD2 配方（验证本套装配与历史一致）
    ("H0_old_DD2_control", OLD_HEAD, OLD_TAIL),
    # H1：旧 DD2 + 新 H94/H95 轻量并入（比 G1 更轻）
    ("H1_old_plus_new05",
     renorm({**OLD_HEAD, "H94": .05, "H95": .05}),
     renorm({**OLD_TAIL, "H94": .05, "H95": .05})),
    # H2：旧 DD2 + 新 H94/H95 较重并入
    ("H2_old_plus_new10",
     renorm({**OLD_HEAD, "H94": .10, "H95": .10}),
     renorm({**OLD_TAIL, "H94": .10, "H95": .10})),
    # H3：只换尾部池为新版（其余全旧）—— 隔离"新池是否有增量"
    ("H3_old_with_newpool",
     OLD_HEAD,
     {"ours": .15, "binary": .10, "oH94": .28, "oH95": .10,
      "nL15": .07, "nL7": .07, "n07s07": .07, "n11s24": .07, "n15s24": .07}),
    # H4：旧 DD2 头部加重旧来源（往 0.8049 之上推）
    ("H4_old_head_heavy",
     {"ours": .13, "binary": .38, "oH94": .245, "oH95": .245},
     OLD_TAIL),
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
        f = OUT / f"submission_v128_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:44s} sha={file_sha256(f)[:16]}", flush=True)

    for tag, h, t in JOBS:
        emit(tag, h, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
