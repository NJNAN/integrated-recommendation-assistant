# -*- coding: utf-8 -*-
"""v127：合并 D2（binary 加重）与 E1（尾部换向）两个已验证方向，并继续推。

线上实测（干净底座）
--------------------
C0 基准                       0.8024  (NDCG .8937  R .4702  GAUC .8773)
D1 binary .48                 0.8025  (+.0001)
D2 binary .52                 0.8027  (+.0003)  ← 头部最佳
D3 ours .20                   0.8024  (±0)
E1 尾部新模型加重             0.8026  (+.0002)  ← 尾部最佳

F 组：把两个方向合并，并继续加推
G 组：把旧 H94/H95 作为**额外来源**并入（新旧 ρ 仅 0.90），用于判断
      "旧来源是否携带干净版没有的信号" —— 仅用于榜单，报告里不得称全干净
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
OUT = ROOT / "outputs/competition_v127_merge_push"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

PATHS = {
    "ours":    ROOT / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary":  ROOT / "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    "H94":     IN / "submission_v120_H94fixed.csv",       # 新
    "H95":     IN / "submission_v120_H95fixed.csv",       # 新
    "oH94":    ROOT / "outputs/competition_v94_O2_H2/submission_v94_H2.csv",           # 旧
    "oH95":    ROOT / "outputs/competition_v95_O2_H2_fixed/submission_v95_H2.csv",     # 旧
    "H2L15":   ROOT / "outputs/competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "H2L7":    ROOT / "outputs/competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "p07s07":  V124 / "submission_v124_H2L07s07.csv",
    "p11s24":  V124 / "submission_v124_H2L11s24.csv",
    "p15s24":  V124 / "submission_v124_H2L15s24.csv",
}
E1_TAIL = {"ours": .10, "binary": .10, "H94": .24, "H95": .10,
           "H2L15": .09, "H2L7": .09, "p07s07": .09, "p11s24": .09, "p15s24": .09}
E1_TAIL_POOL11 = {"ours": .08, "binary": .08, "H94": .18, "H95": .08,
                  "H2L15": .11, "H2L7": .11, "p07s07": .11, "p11s24": .11, "p15s24": .11}

JOBS = [
    # F 组：合并 + 加推（全干净）
    ("F1_D2head_E1tail",  {"ours": .15, "binary": .52, "H94": .165, "H95": .165}, E1_TAIL),
    ("F2_bin56",          {"ours": .15, "binary": .56, "H94": .145, "H95": .145}, E1_TAIL),
    ("F3_bin60",          {"ours": .15, "binary": .60, "H94": .125, "H95": .125}, E1_TAIL),
    ("F4_pool11",         {"ours": .15, "binary": .52, "H94": .165, "H95": .165}, E1_TAIL_POOL11),
    # G 组：并入旧 H94/H95 作为额外来源（榜单用）
    ("G1_add_old_H94H95",
     {"ours": .13, "binary": .36, "oH94": .185, "oH95": .185, "H94": .07, "H95": .07},
     {"ours": .13, "binary": .09, "oH94": .24, "oH95": .09, "H94": .10, "H95": .09,
      "H2L15": .05, "H2L7": .05, "p07s07": .05, "p11s24": .05, "p15s24": .05}),
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
    print(f"载入 {len(S)} 个来源（含旧 H94/H95）", flush=True)

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
        f = OUT / f"submission_v127_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:40s} sha={file_sha256(f)[:16]}", flush=True)

    print("=== F 组：合并 D2+E1，并继续加推（全干净来源）===")
    for tag, h, t in JOBS:
        emit(tag, h, t)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
