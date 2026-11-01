"""v65：在多月监督的基础上放开模型容量，并做 seed 集成。

思路
----
已线上验证：第9组同一套特征，监督从 1 个月 → 3 个月，单独分从 0.7912 → 0.7940。
说明他们的参数配置（num_leaves=31，为其 58k 行单月数据所设）对更大的训练集是欠拟合的。

本脚本在 3 窗口 / 5 窗口两套监督上，扫描容量与轮数：
    leaves ∈ {31, 63, 127}, rounds ∈ {80, 150, 250}
并对较优配置做多种子平均（纯方差削减）。

工程说明
--------
特征构建最贵（每套窗口 ~70s），因此把拼好的训练框与全量打分框缓存到 pkl，
后续重复运行直接复用。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "0.7912的模型/recommendation"
OUT = ROOT / "outputs/competition_v65_capacity"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

WIN3 = [("2010-08", "2010-09-01 16:26:00"),
        ("2010-09", "2010-10-01 16:26:00"),
        ("2010-10", "2010-10-31 16:26:00")]
WIN5 = [("2010-06", "2010-07-01 16:26:00"),
        ("2010-07", "2010-08-01 16:26:00")] + WIN3
CONFIGS = [
    ("L31_r80",    dict(num_leaves=31,  n=80)),
    ("L63_r100",   dict(num_leaves=63,  n=100)),
    ("L63_r200",   dict(num_leaves=63,  n=200)),
    ("L127_r150",  dict(num_leaves=127, n=150)),
]
ENS_SEEDS = [42, 1, 2024]
ENS_CFG = dict(num_leaves=63, n=150)

sys.path.insert(0, str(EXT))
import optimize_v6 as o                     # noqa: E402
o.TRAIN_PATH = str(ROOT / "data/train.csv")
o.CAND_PATH = str(ROOT / "data/sample_candidates.csv")
import final_v10 as f10                     # noqa: E402


def urank(v: np.ndarray, uid: np.ndarray) -> np.ndarray:
    s, g = pd.Series(v), pd.Series(uid)
    return (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()


def build_frames(tag: str, windows, train, cand_pairs, base_feats, t0):
    fp = CACHE / f"train_{tag}.pkl"
    fq = CACHE / f"full.pkl"
    if fp.exists() and fq.exists():
        print(f"  [{tag}] cache hit", flush=True)
        return pd.read_pickle(fp), pd.read_pickle(fq)
    frames = []
    for name, T in windows:
        T = pd.Timestamp(T)
        df = f10.build_feature_df(train[train["InvoiceDate"] < T], cand_pairs, cutoff_days=30)
        df["_window"] = name
        print(f"  [{tag}] window {name}: rows={len(df)} pos={int(df['label'].sum())} "
              f"({time.time()-t0:.0f}s)", flush=True)
        frames.append(df)
    tr = pd.concat(frames, ignore_index=True)
    if not fq.exists():
        pd.DataFrame(f10.build_feature_df(train, cand_pairs, cutoff_days=None)).to_pickle(fq)
    pd.DataFrame(tr).to_pickle(fp)
    return tr, pd.read_pickle(fq)


def main() -> None:
    t0 = time.time()
    train, cd = o.load_data()
    cand_pairs = cd[["user_id", "item_id"]].drop_duplicates().copy()
    base_feats = o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                            "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                            "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                            "rep_cnt_rank", "item_total_orders_rank"]
    uid = cand_pairs["user_id"].to_numpy()
    v9_rank = f10.v9_score(train, cand_pairs)
    meta = {"outputs": [], "configs": [], "note": "多月监督 + 容量扫描 + seed 集成"}

    for tag, windows in [("win3", WIN3), ("win5", WIN5)]:
        tr, full = build_frames(tag, windows, train, cand_pairs, base_feats, t0)
        X = tr[base_feats].astype(float)
        y = tr["label"].values
        grp = tr.groupby(["_window", "user_id"], sort=False).size().values
        Xfu = full[base_feats].astype(float)
        print(f"  [{tag}] combined rows={len(tr)} pos={int(tr['label'].sum())} "
              f"({time.time()-t0:.0f}s)", flush=True)

        def emit(name: str, rk: np.ndarray) -> None:
            for suffix, final in [("ranker", rk),
                                  ("v9blend", f10.RANKER_W * rk + (1 - f10.RANKER_W) * v9_rank)]:
                sub = pd.DataFrame({"user_id": cand_pairs["user_id"].to_numpy(),
                                    "item_id": cand_pairs["item_id"].to_numpy(),
                                    "score": final})
                f = OUT / f"submission_v65_{tag}_{name}_{suffix}.csv"
                sub.to_csv(f, index=False, float_format="%.6f")
                meta["outputs"].append(f.name)
            print(f"  saved {tag}/{name} ({time.time()-t0:.0f}s)", flush=True)

        for name, cfg in CONFIGS:
            params = {
                "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
                "num_leaves": cfg["num_leaves"], "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_data_in_leaf": 100, "verbose": -1, "seed": 42,
                "lambda_l1": 1.0, "lambda_l2": 10.0,
            }
            ds = lgb.Dataset(X, label=y, group=grp)
            m = lgb.train(params, ds, num_boost_round=cfg["n"])
            emit(name, urank(m.predict(Xfu), uid))

        # seed 集成（纯方差削减）
        acc = []
        for sd in ENS_SEEDS:
            params = {
                "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
                "num_leaves": ENS_CFG["num_leaves"], "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_data_in_leaf": 100, "verbose": -1, "seed": sd,
                "lambda_l1": 1.0, "lambda_l2": 10.0,
            }
            ds = lgb.Dataset(X, label=y, group=grp)
            m = lgb.train(params, ds, num_boost_round=ENS_CFG["n"])
            acc.append(urank(m.predict(Xfu), uid))
        emit("ENS3", np.mean(acc, axis=0))

    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
