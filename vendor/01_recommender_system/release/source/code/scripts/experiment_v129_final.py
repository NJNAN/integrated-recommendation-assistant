# -*- coding: utf-8 -*-
"""v129：修正后的最终提交集。

背景修正（三处，全部已落地）
---------------------------
1. v120 现在传了 candidate_frame → 7 个候选统计特征恢复（实测非零 20 万+ 行）。
2. v120b 拆成三类产物：
     H2f_r120_s42  120轮/seed42  = 与旧 H94/H95 唯一差别是 Bug 修复 → 公平对比用
     H2f_r300_s42  300轮/seed42  = 新模型（含容量改动）
     H2f_r300_s07  300轮/seed7   = 新模型（容量+种子）
   装配别名 H94fixed/H95fixed = 后两者。
3. ours（含未修的 v36）无法重训，本组额外给出**剔除 ours** 的版本，
   用于回答「把自认脏的 5.25% 完全去掉后能到几分」。
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
OUT = ROOT / "outputs/competition_v129_final"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.train_competition import file_sha256                       # noqa: E402
from scripts.make_submission_v11_season import (                        # noqa: E402
    read_candidates, validate_submission,
)

PATHS = {
    "ours":   O / "competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
    "binary": O / "competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
    # 旧（带病管线）
    "oH94":   O / "competition_v94_O2_H2/submission_v94_H2.csv",
    "oH95":   O / "competition_v95_O2_H2_fixed/submission_v95_H2.csv",
    "oL15":   O / "competition_v102_H2_variants/submission_v102_H2L15.csv",
    "oL7":    O / "competition_v104_smallcap/submission_v104_H2L7.csv",
    "o07":    O / "competition_v106_headfocus/submission_v106_H2L07s07.csv",
    "o11":    O / "competition_v106_headfocus/submission_v106_H2L11s24.csv",
    "o15":    O / "competition_v106_headfocus/submission_v106_H2L15s24.csv",
    # 新（修复管线，已含候选统计）
    "H94":    IN / "submission_v120_H94fixed.csv",
    "H95":    IN / "submission_v120_H95fixed.csv",
    "nL15":   O / "competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
    "nL7":    O / "competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
    "n07":    V124 / "submission_v124_H2L07s07.csv",
    "n11":    V124 / "submission_v124_H2L11s24.csv",
    "n15":    V124 / "submission_v124_H2L15s24.csv",
}
OLD_HEAD = {"ours": .15, "binary": .42, "oH94": .215, "oH95": .215}
OLD_TAIL = {"ours": .15, "binary": .10, "oH94": .28, "oH95": .10,
            "oL15": .07, "oL7": .07, "o07": .07, "o11": .07, "o15": .07}
CLEAN_HEAD = {"ours": .15, "binary": .52, "H94": .165, "H95": .165}
CLEAN_TAIL = {"ours": .10, "binary": .10, "H94": .24, "H95": .10,
              "nL15": .09, "nL7": .09, "n07": .09, "n11": .09, "n15": .09}
NOOURS_HEAD = {"binary": .61, "H94": .195, "H95": .195}
NOOURS_TAIL = {"binary": .11, "H94": .27, "H95": .11,
               "nL15": .10, "nL7": .10, "n07": .10, "n11": .10, "n15": .10}


def renorm(w, total=1.0):
    s = sum(w.values())
    return {k: v * total / s for k, v in w.items() if v}


JOBS = [
    ("CLEAN_D2E1", CLEAN_HEAD, CLEAN_TAIL),
    ("CLEAN_no_ours", NOOURS_HEAD, NOOURS_TAIL),
    ("MIX_new05", renorm({**OLD_HEAD, "H94": .05, "H95": .05}),
                  renorm({**OLD_TAIL, "H94": .05, "H95": .05})),
    ("MIX_new10", renorm({**OLD_HEAD, "H94": .10, "H95": .10}),
                  renorm({**OLD_TAIL, "H94": .10, "H95": .10})),
    ("MIX_newpool", OLD_HEAD,
                    {"ours": .15, "binary": .10, "oH94": .28, "oH95": .10,
                     "nL15": .07, "nL7": .07, "n07": .07, "n11": .07, "n15": .07}),
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
        f = OUT / f"submission_v129_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"  {f.name:40s} sha={file_sha256(f)[:16]}", flush=True)

    print("=== 修正后的最终提交集 ===")
    for tag, h, t in JOBS:
        emit(tag, h, t)

    # 公平复刻单模型：直接复制 v120b 的 r120 产物
    import shutil
    src = IN / "submission_v120_H2f_r120_s42.csv"
    dst = OUT / "submission_v129_FAIR_H2_r120.csv"
    shutil.copyfile(src, dst)
    print(f"  {dst.name:40s} sha={file_sha256(dst)[:16]}  ← 公平复刻（对比旧 H94=0.8007）", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
