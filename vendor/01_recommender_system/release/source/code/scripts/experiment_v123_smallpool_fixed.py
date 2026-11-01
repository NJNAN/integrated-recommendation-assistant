"""v105：验证「尾部需要多个不同小模型投票」假设。

线上 GAUC 曲线（容量轴，头部固定为 AV5）：
    基准 0.8779 / L3 0.8781 / L5 0.8785 / L15 0.8785 / L7 0.8788
    L15+L5 0.8788 / L15+L7 0.8789（最佳，综合 0.8048）
假设：谷底在 L7~L15 之间，且**叠加多个不同小容量**比单个更好。
本脚本训练 L9/L11，然后测「小模型平均池」是否优于最优单变体。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.experiment_v94_O2_H2 as v94  # noqa: E402
from scripts.fixed_alignment import (  # noqa: E402
    load_aligned_nov, load_aligned_train,
)
from scripts.train_competition import file_sha256  # noqa: E402
from scripts.make_submission_v11_season import read_candidates, validate_submission  # noqa: E402

CACHE = ROOT / "outputs/competition_v65_capacity/cache"
IN = ROOT / "outputs/competition_v120_H2_fixed"
V102 = ROOT / "outputs/competition_v102_H2_variants"
V104 = ROOT / "outputs/competition_v104_smallcap"
OUT = ROOT / "outputs/competition_v123_smallpool_fixed"
OUT.mkdir(parents=True, exist_ok=True)
NEW = {"H2L9": 9, "H2L11": 11}
BASE = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
        "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "min_data_in_leaf": 100, "verbose": -1, "seed": 42,
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
    # [修复] 按键校验对齐，禁止纯行号拼接
    _ours_tr, _sig_tr = load_aligned_train(tr, IN, v94.SIGNAL_COLS)
    _ours_fu, _sig_fu = load_aligned_nov(full, IN, cand_pairs, v94.SIGNAL_COLS)
    Xtr = pd.concat([tr[base_feats].astype(float), _ours_tr,
                     _sig_tr[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float), _ours_fu,
                     _sig_fu[v94.SIGNAL_COLS]], axis=1)
    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values
    for tag, nl in NEW.items():
        m = lgb.train({**BASE, "num_leaves": nl}, lgb.Dataset(Xtr, label=y, group=grp),
                      num_boost_round=300)
        s, g = pd.Series(m.predict(Xfu)), pd.Series(uid)
        rk = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
        f = OUT / f"submission_v123_{tag}.csv"
        pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                      "item_id": cand_pairs.item_id.to_numpy(),
                      "score": rk}).to_csv(f, index=False, float_format="%.6f")
        print(f"  {tag:8s} sha={file_sha256(f)[:16]}  ({time.time()-t0:.0f}s)", flush=True)

    paths = {
        "ours": ROOT / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
        "binary": ROOT / "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
        "H94": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H94fixed.csv",
        "H95": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H95fixed.csv",
        "H2L15": V102 / "submission_v102_H2L15.csv",
        "H2L7": V104 / "submission_v104_H2L7.csv",
        "H2L5": V104 / "submission_v104_H2L5.csv",
    }
    for tag in NEW:
        paths[tag] = OUT / f"submission_v123_{tag}.csv"
    S = {}
    for n, p in paths.items():
        d = pd.read_csv(p, dtype={"user_id": str, "item_id": str})
        d["k"] = d.user_id + "|" + d.item_id
        x = d.set_index("k").score.reindex(cand_pairs.k).to_numpy(float)
        s, g = pd.Series(x), pd.Series(uid)
        S[n] = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()

    def emit(tag, head, tail, cut=10):
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
    print("=== CC：小模型投票池 vs 最优单变体 ===")
    # CC0：复刻当前最佳 CB_L15_L7（对照，应 = 0.8048）
    emit("CC0_ref_CB_L15_L7", HEAD,
         {"ours": .15, "binary": .10, "H94": .30, "H2L15": .15, "H2L7": .15, "H95": .15})
    # CC1：L15+L11+L7 三小模型
    emit("CC1_L15_L11_L7", HEAD,
         {"ours": .15, "binary": .10, "H94": .30, "H2L15": .125, "H2L11": .125, "H2L7": .125, "H95": .125})
    # CC2：四个小模型等权（L15/L11/L9/L7）
    emit("CC2_pool4", HEAD,
         {"ours": .15, "binary": .10, "H94": .30, "H2L15": .09, "H2L11": .09, "H2L9": .09, "H2L7": .09, "H95": .13})
    # CC3：小模型池替代 H95 的一半，H94 权重也降
    emit("CC3_pool_lean", HEAD,
         {"ours": .15, "binary": .10, "H94": .25, "H2L15": .12, "H2L11": .12, "H2L9": .12, "H2L7": .12, "H95": .10})
    # CC4：L11 单嵌（补全容量曲线：L7/L11 之间是否更优）
    emit("CC4_L11_tail", HEAD,
         {"ours": .15, "binary": .10, "H94": .375, "H2L11": .1875, "H95": .1875})
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
