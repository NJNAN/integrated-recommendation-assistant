"""v104：沿「尾部小容量」方向继续推——训练 H2L7 / H2L5 / H2L3，并出尾部嵌入候选。

依据（线上实测，全部 NDCG 0.8962 不变、只有 GAUC 变）：
    H2L15  -> GAUC 0.8785   综合 0.8046   ← 当前最佳
    H2S2024-> GAUC 0.8779   综合 0.8044
    H2BIN  -> GAUC 0.8777   综合 0.8043
    H2L127 -> GAUC 0.8774   综合 0.8042
    AV5 基准-> GAUC 0.8779  综合 0.8044
头部嵌入（EA_*）全部掉分，所以本脚本只动尾部。

安全：特征/标签仍是 5 窗口，11 月预测特征来自 train.csv（<2010-11-01）。
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
import scripts.experiment_v94_O2_H2 as v94  # noqa: E402
from scripts.train_competition import file_sha256  # noqa: E402
from scripts.make_submission_v11_season import read_candidates, validate_submission  # noqa: E402

CACHE = ROOT / "outputs/competition_v65_capacity/cache"
IN = ROOT / "outputs/competition_v95_O2_H2_fixed"
V102 = ROOT / "outputs/competition_v102_H2_variants"
OUT = ROOT / "outputs/competition_v104_smallcap"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
ROUNDS = 300
SMALL = {"H2L7": 7, "H2L5": 5, "H2L3": 3}
BASE = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
        "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "min_data_in_leaf": 100, "verbose": -1, "seed": SEED,
        "lambda_l1": 1.0, "lambda_l2": 10.0}


def main() -> None:
    t0 = time.time()
    cand_pairs = pd.read_csv(ROOT / "data/sample_candidates.csv",
                             dtype={"user_id": str, "item_id": str})
    uid = cand_pairs.user_id.to_numpy()
    cand_pairs["k"] = cand_pairs.user_id + "|" + cand_pairs.item_id
    tr = pd.read_pickle(CACHE / "train_win5.pkl")
    full = pd.read_pickle(CACHE / "full.pkl")
    base_feats = v94.o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                                "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                                "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                                "rep_cnt_rank", "item_total_orders_rank"]
    Xtr = pd.concat([tr[base_feats].astype(float),
                     pd.read_pickle(IN / "our_feats_train.pkl").astype(float),
                     pd.read_pickle(IN / "signals_train.pkl")[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float),
                     pd.read_pickle(IN / "our_feats_nov.pkl").astype(float),
                     pd.read_pickle(IN / "signals_nov.pkl")[v94.SIGNAL_COLS]], axis=1)
    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values

    # 训练小容量变体
    for tag, nl in SMALL.items():
        m = lgb.train({**BASE, "num_leaves": nl}, lgb.Dataset(Xtr, label=y, group=grp),
                      num_boost_round=ROUNDS)
        s, g = pd.Series(m.predict(Xfu)), pd.Series(uid)
        rk = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
        f = OUT / f"submission_v104_{tag}.csv"
        pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                      "item_id": cand_pairs.item_id.to_numpy(),
                      "score": rk}).to_csv(f, index=False, float_format="%.6f")
        print(f"  {tag:8s} sha={file_sha256(f)[:16]}  ({time.time()-t0:.0f}s)", flush=True)

    # 装配尾部嵌入候选
    paths = {
        "ours": ROOT / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
        "binary": ROOT / "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
        "H94": ROOT / "outputs/competition_v94_O2_H2/submission_v94_H2.csv",
        "H95": ROOT / "outputs/competition_v95_O2_H2_fixed/submission_v95_H2.csv",
        "L15": V102 / "submission_v102_H2L15.csv",
    }
    for tag in SMALL:
        paths[tag] = OUT / f"submission_v104_{tag}.csv"
    S = {}
    for n, p in paths.items():
        d = pd.read_csv(p, dtype={"user_id": str, "item_id": str})
        d["k"] = d.user_id + "|" + d.item_id
        x = d.set_index("k").score.reindex(cand_pairs.k).to_numpy(float)
        s, g = pd.Series(x), pd.Series(uid)
        S[n] = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()

    def emit(tag: str, head: dict, tail: dict, cut: int = 10) -> None:
        remaining = pd.DataFrame({"u": np.asarray(uid), "k": cand_pairs.k.to_numpy()})
        picked = []
        for i, wt in enumerate([head, tail]):
            sc = sum(w * S[n] for n, w in wt.items())
            remaining["sc"] = sc[remaining.index.to_numpy()]
            ordered = remaining.sort_values(["u", "sc"], ascending=[True, False], kind="mergesort")
            part = ordered if i == 1 else ordered.groupby("u", sort=False).head(cut)
            part = part.copy(); part["_piece"] = i
            picked.append(part.drop(columns="sc"))
            remaining = remaining.drop(index=part.index)
        assert len(remaining) == 0
        fin = (pd.concat(picked, ignore_index=True)
               .sort_values(["u", "_piece"], kind="mergesort").reset_index(drop=True))
        fin["score"] = -fin.groupby("u").cumcount()
        sc = fin.set_index("k").score.reindex(cand_pairs.k).to_numpy(float)
        sub = pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                            "item_id": cand_pairs.item_id.to_numpy(), "score": sc})
        validate_submission(cand_pairs, sub)
        f = OUT / f"submission_{tag}.csv"
        sub.to_csv(f, index=False, float_format="%.6f")
        validate_submission(cand_pairs, pd.read_csv(f, dtype={"user_id": str, "item_id": str}))
        print(f"{f.name:46s} sha={file_sha256(f)[:16]}")

    HEAD = {"ours": .15, "binary": .42, "H94": .215, "H95": .215}
    print("=== CA：尾部嵌入更小容量 ===")
    for tag in SMALL:
        emit(f"CA_{tag}_tail", HEAD,
             {"ours": .15, "binary": .10, "H94": .375, tag: .1875, "H95": .1875})
    print("=== CB：L15 + 小容量双嵌入 ===")
    emit("CB_L15_L7", HEAD, {"ours": .15, "binary": .10, "H94": .30, "L15": .15, "H2L7": .15, "H95": .15})
    emit("CB_L15_x2", HEAD, {"ours": .15, "binary": .10, "H94": .30, "H95": .20, "L15": .25})
    emit("CB_L15_L5", HEAD, {"ours": .15, "binary": .10, "H94": .30, "L15": .15, "H2L5": .15, "H95": .15})
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
