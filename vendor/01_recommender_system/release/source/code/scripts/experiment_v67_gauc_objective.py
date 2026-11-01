"""v67：针对 GAUC 的排序目标改造。

依据（全部来自线上实测）
------------------------
1. 融合增益按指标分解：NDCG 存在「共识天花板」≈0.887——把伙伴从 v10(0.8803)
   换成更強的 v61(0.8837)，融合 NDCG 反而略降(0.8875→0.8869)；
2. 但 GAUC 会从伙伴完整传递到融合：v10 0.8657→融合 0.8700，v61 0.8690→融合 0.8724；
3. 第9组管线的 GAUC 短板有明确技术原因：lambdarank 的
   lambdarank_truncation_level 默认 30，只对每个用户前 30 名做 pairwise 优化，
   30 名之后的排序（候选数中位数 100、最大 344）没有梯度。GAUC 是全排序指标，
   尾部恰恰是它的计分区。

因此本脚本在**同一套特征与监督**上只改排序目标：
    a) lambdarank, truncation_level=1000   —— 全 pair 优化
    b) lambdarank, truncation_level=120    —— 折中
    c) binary(label)                        —— 逐样本对数损失，优化整体校准
    d) rank_xendcg                          —— 另一种排序损失
对照为原配置（truncation 默认 30）。

目的不是让单模型总分更高，而是造出**GAUC 更高**的融合伙伴。
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
OUT = ROOT / "outputs/competition_v67_gauc_objective"
CACHE_IN = ROOT / "outputs/competition_v65_capacity/cache"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = [("2010-06", "2010-07-01 16:26:00"), ("2010-07", "2010-08-01 16:26:00"),
           ("2010-08", "2010-09-01 16:26:00"), ("2010-09", "2010-10-01 16:26:00"),
           ("2010-10", "2010-10-31 16:26:00")]
ROUNDS = 120
SEED = 42

sys.path.insert(0, str(EXT))
import optimize_v6 as o                     # noqa: E402
o.TRAIN_PATH = str(ROOT / "data/train.csv")
o.CAND_PATH = str(ROOT / "data/sample_candidates.csv")
import final_v10 as f10                     # noqa: E402


def urank(v: np.ndarray, uid: np.ndarray) -> np.ndarray:
    s, g = pd.Series(v), pd.Series(uid)
    return (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()


def main() -> None:
    t0 = time.time()
    train, cd = o.load_data()
    cand_pairs = cd[["user_id", "item_id"]].drop_duplicates().copy()
    base_feats = o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                            "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                            "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                            "rep_cnt_rank", "item_total_orders_rank"]
    uid = cand_pairs["user_id"].to_numpy()

    fp, fq = CACHE_IN / "train_win5.pkl", CACHE_IN / "full.pkl"
    if fp.exists() and fq.exists():
        tr, full = pd.read_pickle(fp), pd.read_pickle(fq)
        print(f"  cache hit ({time.time()-t0:.0f}s)", flush=True)
    else:
        frames = []
        for name, T in WINDOWS:
            T = pd.Timestamp(T)
            df = f10.build_feature_df(train[train["InvoiceDate"] < T], cand_pairs, cutoff_days=30)
            df["_window"] = name
            print(f"  window {name}: rows={len(df)} pos={int(df['label'].sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            frames.append(df)
        tr = pd.concat(frames, ignore_index=True)
        full = f10.build_feature_df(train, cand_pairs, cutoff_days=None)
        # 缓存给后续脚本复用
        CACHE_IN.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tr).to_pickle(fp)
        pd.DataFrame(full).to_pickle(fq)

    X = tr[base_feats].astype(float)
    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values
    Xfu = full[base_feats].astype(float)
    v9_rank = f10.v9_score(train, cand_pairs)
    print(f"  rows={len(tr)} pos={int(y.sum())} rounds={ROUNDS} ({time.time()-t0:.0f}s)", flush=True)

    BASE = {"num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "min_data_in_leaf": 100,
            "verbose": -1, "seed": SEED, "lambda_l1": 1.0, "lambda_l2": 10.0}

    VARIANTS = {
        "t30":   {**BASE, "objective": "lambdarank", "metric": "ndcg",
                  "ndcg_eval_at": [5, 10, 20, 1000], "lambdarank_truncation_level": 30},
        "t120":  {**BASE, "objective": "lambdarank", "metric": "ndcg",
                  "ndcg_eval_at": [5, 10, 20, 1000], "lambdarank_truncation_level": 120},
        "t1000": {**BASE, "objective": "lambdarank", "metric": "ndcg",
                  "ndcg_eval_at": [5, 10, 20, 1000], "lambdarank_truncation_level": 1000},
        "binary": {**BASE, "objective": "binary", "metric": "auc"},
        "xendcg": {**BASE, "objective": "rank_xendcg", "metric": "ndcg",
                   "ndcg_eval_at": [5, 10, 20, 1000]},
    }

    meta = {"rounds": ROUNDS, "variants": list(VARIANTS), "outputs": [],
            "note": "同一特征与监督，仅改排序目标；用于造 GAUC 更高的融合伙伴。"}
    for tag, params in VARIANTS.items():
        need_group = params["objective"] in ("lambdarank", "rank_xendcg")
        ds = lgb.Dataset(X, label=y, group=grp if need_group else None)
        m = lgb.train(params, ds, num_boost_round=ROUNDS)
        rk = urank(m.predict(Xfu), uid)
        for suffix, final in [("ranker", rk),
                              ("v9blend", f10.RANKER_W * rk + (1 - f10.RANKER_W) * v9_rank)]:
            sub = pd.DataFrame({"user_id": cand_pairs["user_id"].to_numpy(),
                                "item_id": cand_pairs["item_id"].to_numpy(), "score": final})
            f = OUT / f"submission_v67_{tag}_{suffix}.csv"
            sub.to_csv(f, index=False, float_format="%.6f")
            meta["outputs"].append(f.name)
        print(f"  saved v67/{tag} ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
