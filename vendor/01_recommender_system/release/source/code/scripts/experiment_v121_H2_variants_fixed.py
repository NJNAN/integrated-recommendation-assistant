"""v102：H2 家族的「强变体」——同一个 141 特征矩阵，换容量 / 目标 / 种子。

动机
----
H94+H95 双变体把分段融合推高了 +0.0007，但两者只差一个特征（相关度 0.979）。
「同族强变体叠加」这条规律还没在**容量轴**和**目标轴**上验证过。
特征矩阵已由 v95 缓存，所以这里只训练，不重算特征。

安全性
------
特征与标签仍是 5 窗口（2010-06..10），特征重建的 history 一律 < 该窗口 cutoff；
11 月预测特征用全量 train.csv（max < 2010-11-01，脚本内断言）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.experiment_v94_O2_H2 as v94  # noqa: E402  复用缓存路径与列定义
from scripts.fixed_alignment import (  # noqa: E402
    load_aligned_nov, load_aligned_train,
)
from scripts.train_competition import file_sha256  # noqa: E402

CACHE = ROOT / "outputs/competition_v65_capacity/cache"
IN = ROOT / "outputs/competition_v120_H2_fixed"
OUT = ROOT / "outputs/competition_v121_H2_variants_fixed"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
ROUNDS = 300


def main() -> None:
    t0 = time.time()
    cand_pairs = pd.read_csv(ROOT / "data/sample_candidates.csv",
                             dtype={"user_id": str, "item_id": str})
    uid = cand_pairs.user_id.to_numpy()
    tr = pd.read_pickle(CACHE / "train_win5.pkl")
    full = pd.read_pickle(CACHE / "full.pkl")
    base_feats = v94.o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                                "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                                "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                                "rep_cnt_rank", "item_total_orders_rank"]
    # [修复] 按键校验对齐，禁止纯行号拼接
    OUR_TR, SIG_TR = load_aligned_train(tr, IN, v94.SIGNAL_COLS)
    nov_ours, sig_nov = load_aligned_nov(full, IN, cand_pairs, v94.SIGNAL_COLS)
    Xtr = pd.concat([tr[base_feats].astype(float), OUR_TR,
                     SIG_TR[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float), nov_ours,
                     sig_nov[v94.SIGNAL_COLS]], axis=1)
    assert list(Xtr.columns) == list(Xfu.columns)
    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values
    print(f"X {Xtr.shape} / {Xfu.shape}  正例率 {y.mean():.4f}", flush=True)

    BASE = {"metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
            "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "min_data_in_leaf": 100,
            "verbose": -1, "seed": SEED, "lambda_l1": 1.0, "lambda_l2": 10.0}
    VARIANTS = {
        "H2L15": {"objective": "lambdarank", "num_leaves": 15},
        "H2L31": {"objective": "lambdarank", "num_leaves": 31},
        "H2L127": {"objective": "lambdarank", "num_leaves": 127},
        "H2T1000": {"objective": "lambdarank", "num_leaves": 63, "lambdarank_truncation_level": 1000},
        "H2BIN": {"objective": "binary", "num_leaves": 63},
        "H2S2024": {"objective": "lambdarank", "num_leaves": 63, "seed": 2024},
    }
    meta = {"outputs": []}
    for tag, over in VARIANTS.items():
        p = {**BASE, **over}
        if p["objective"] == "binary":
            p.pop("ndcg_eval_at", None)
            p["metric"] = "auc"
            ds = lgb.Dataset(Xtr, label=y)
            rounds = 400
        else:
            ds = lgb.Dataset(Xtr, label=y, group=grp)
            rounds = ROUNDS
        m = lgb.train(p, ds, num_boost_round=rounds)
        s, g = pd.Series(m.predict(Xfu)), pd.Series(uid)
        rk = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
        f = OUT / f"submission_v121_{tag}.csv"
        pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                      "item_id": cand_pairs.item_id.to_numpy(),
                      "score": rk}).to_csv(f, index=False, float_format="%.6f")
        pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(
            ascending=False).to_csv(OUT / f"importance_{tag}.csv")
        meta["outputs"].append({"tag": tag, "file": f.name, "sha": file_sha256(f)[:16]})
        print(f"  {tag:10s} sha={file_sha256(f)[:16]}  ({time.time()-t0:.0f}s)", flush=True)
    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
