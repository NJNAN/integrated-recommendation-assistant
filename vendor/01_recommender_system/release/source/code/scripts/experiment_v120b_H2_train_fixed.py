# -*- coding: utf-8 -*-
"""v120b：用 v120 修复后的特征训练 H94 / H95 的对应版本。

为什么需要这一步
----------------
v95 原本同时做两件事：(1) 生成 5 窗口特征；(2) 用这些特征训练 O2 与 H2，
产出 `submission_v95_O2.csv` / `submission_v95_H2.csv`（即 DD2 里的 H95）。
v120 只复刻了第 (1) 件事——修好了引擎穿越与行错位，但没有产出模型。
因此 DD2 的装配一直在拿**旧口径训练出来的 H94/H95**。

本脚本补上第 (2) 件事，并**明确区分两类产物**（不要混为一谈）：

    r120_s42  ← 忠实复刻：轮数/种子与旧 H94/H95 完全一致（120 轮 / seed 42），
                 与旧版唯一差别就是特征口径的 Bug 修复。
                 **只有它能用于「干净 vs 带病」的公平对比。**
    r300_s42  ← 变量：把轮数提到 300（容量轴的一次改动，不是纯 Bug 修复）
    r300_s07  ← 变量：300 轮 + seed 7

装配用文件：
    submission_v120_H94fixed.csv = r300_s42
    submission_v120_H95fixed.csv = r300_s07
（这两个是"新模型"，不得在报告里声称与旧版仅差 Bug 修复。）
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "0.7912的模型/recommendation"
CACHE = ROOT / "outputs/competition_v65_capacity/cache"
IN = ROOT / "outputs/competition_v120_H2_fixed"
OUT = ROOT / "outputs/competition_v120_H2_fixed"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))
import optimize_v6 as o                                                    # noqa: E402
o.TRAIN_PATH = str(ROOT / "data/train.csv")
o.CAND_PATH = str(ROOT / "data/sample_candidates.csv")

import scripts.experiment_v94_O2_H2 as v94                                 # noqa: E402
from scripts.fixed_alignment import load_aligned_nov, load_aligned_train   # noqa: E402
from scripts.train_competition import file_sha256                          # noqa: E402
from scripts.make_submission_v11_season import (                           # noqa: E402
    read_candidates, validate_submission,
)

BASE = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
        "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_data_in_leaf": 100,
        "verbose": -1, "lambda_l1": 1.0, "lambda_l2": 10.0}
# 旧 v94/v95 的原始配置：ROUNDS=120, SEED=42（见 scripts/experiment_v94_O2_H2.py:40-41）
ROUNDS_FAITHFUL = 120
SEED_FAITHFUL = 42
RUNS = [
    # (输出名, 轮数, 种子, 是否是公平复刻)
    ("H2f_r120_s42", ROUNDS_FAITHFUL, SEED_FAITHFUL, True),
    ("H2f_r300_s42", 300, SEED_FAITHFUL, False),
    ("H2f_r300_s07", 300, 7, False),
]
BASE_FEATS = v94.o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                            "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                            "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                            "rep_cnt_rank", "item_total_orders_rank"]


def main() -> None:
    t0 = time.time()
    cand_pairs = read_candidates(ROOT / "data/sample_candidates.csv")
    uid = cand_pairs.user_id.to_numpy()

    tr = pd.read_pickle(CACHE / "train_win5.pkl")
    full = pd.read_pickle(CACHE / "full.pkl")

    # 键校验对齐（与下游四个脚本同一套工具）
    ours_tr, sig_tr = load_aligned_train(tr, IN, v94.SIGNAL_COLS)
    ours_fu, sig_fu = load_aligned_nov(full, IN, cand_pairs, v94.SIGNAL_COLS)

    Xtr = pd.concat([tr[BASE_FEATS].astype(float), ours_tr, sig_tr[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[BASE_FEATS].astype(float), ours_fu, sig_fu[v94.SIGNAL_COLS]], axis=1)
    assert list(Xtr.columns) == list(Xfu.columns)
    assert len(Xtr) == 291025 and len(Xfu) == 58205

    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values
    print(f"X {Xtr.shape} / {Xfu.shape}  正例率 {y.mean():.4f}  特征列 {len(Xtr.columns)}", flush=True)

    for tag, rounds, seed, faithful in RUNS:
        m = lgb.train({**BASE, "seed": seed}, lgb.Dataset(Xtr, label=y, group=grp),
                      num_boost_round=rounds)
        s, g = pd.Series(m.predict(Xfu)), pd.Series(uid)
        rk = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
        sub = pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                            "item_id": cand_pairs.item_id.to_numpy(), "score": rk})
        validate_submission(cand_pairs, sub)
        f = OUT / f"submission_v120_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        note = "  ← 公平复刻（与旧 H94/H95 同轮数同种子）" if faithful else "  ← 新模型（含容量/种子改动）"
        print(f"  {tag:14s} 轮={rounds:3d} 种子={seed:2d} sha={file_sha256(f)[:16]}{note}", flush=True)
        # 装配用文件：H94fixed ← r300_s42；H95fixed ← r300_s07
        if tag == "H2f_r300_s42":
            alias = OUT / "submission_v120_H94fixed.csv"
        elif tag == "H2f_r300_s07":
            alias = OUT / "submission_v120_H95fixed.csv"
        else:
            alias = None
        if alias is not None:
            alias.write_bytes(f.read_bytes())
            print(f"    -> 装配别名 {alias.name}（sha 相同 {file_sha256(alias)[:16]}）", flush=True)

    print(f"DONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
