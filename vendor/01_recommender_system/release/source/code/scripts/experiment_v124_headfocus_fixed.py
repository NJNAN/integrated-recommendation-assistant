"""v106：两条新轴——头部聚焦（truncation=10/5/20）+ 尾部投票池扩容（多容量×多种子）。

依据
----
① 头部：NDCG@10 自 AU1 起冻结在 0.8962，六条 EA_* 替换实验全部掉分。
   但从未试过「让训练目标只盯前 10 名」——lambdarank 的 truncation_level
   之前只扫过 30/120/1000，**没扫过 10 以下**。
② 尾部：CC1/CC2 证明「多个不同小模型投票」把 GAUC 推到 0.8790（历史最高），
   但池里只有 4 个同种子(num_leaves 7/9/11/15)模型。种子轴尚未进池。

安全：5 窗口标签，11 月预测特征来自 train.csv（<2010-11-01），与之前一致。
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
V105 = ROOT / "outputs/competition_v105_smallpool"
OUT = ROOT / "outputs/competition_v124_headfocus_fixed"
OUT.mkdir(parents=True, exist_ok=True)

BASE = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
        "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "min_data_in_leaf": 100, "verbose": -1, "seed": 42,
        "lambda_l1": 1.0, "lambda_l2": 10.0}
FOCUS = {  # 头部聚焦：梯度只看前 N 名
    "H2T010": {"lambdarank_truncation_level": 10},
    "H2T020": {"lambdarank_truncation_level": 20},
    "H2T005": {"lambdarank_truncation_level": 5},
}
POOL = {  # 尾部投票池扩容：多容量 × 多种子
    "H2L07s24": {"num_leaves": 7, "seed": 2024},
    "H2L11s24": {"num_leaves": 11, "seed": 2024},
    "H2L15s24": {"num_leaves": 15, "seed": 2024},
    "H2L11s07": {"num_leaves": 11, "seed": 7},
    "H2L07s07": {"num_leaves": 7, "seed": 7},
}


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

    def train(tag: str, over: dict) -> None:
        m = lgb.train({**BASE, **over}, lgb.Dataset(Xtr, label=y, group=grp),
                      num_boost_round=300)
        s, g = pd.Series(m.predict(Xfu)), pd.Series(uid)
        rk = (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()
        f = OUT / f"submission_v124_{tag}.csv"
        pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                      "item_id": cand_pairs.item_id.to_numpy(),
                      "score": rk}).to_csv(f, index=False, float_format="%.6f")
        print(f"  {tag:10s} sha={file_sha256(f)[:16]}  ({time.time()-t0:.0f}s)", flush=True)

    print("=== 训练：头部聚焦 + 尾部池扩容 ===")
    for tag, over in {**FOCUS, **POOL}.items():
        train(tag, over)

    # 装配
    paths = {
        "ours": ROOT / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
        "binary": ROOT / "outputs/competition_v67_gauc_objective/submission_v67_binary_v9blend.csv",
        "H94": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H94fixed.csv",
        "H95": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H95fixed.csv",
        "H2L15": ROOT / "outputs/competition_v121_H2_variants_fixed/submission_v121_H2L15.csv",
        "H2L7": ROOT / "outputs/competition_v122_smallcap_fixed/submission_v122_H2L7.csv",
        "H2L5": V104 / "submission_v104_H2L5.csv",
        "H2L9": V105 / "submission_v105_H2L9.csv",
        "H2L11": V105 / "submission_v105_H2L11.csv",
    }
    for tag in {**FOCUS, **POOL}:
        paths[tag] = OUT / f"submission_v124_{tag}.csv"
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
    TAIL_BEST = {"ours": .15, "binary": .10, "H94": .30, "H2L15": .15, "H2L7": .15, "H95": .15}
    print("=== DA：头部换成「聚焦前10名」的模型 ===")
    for tag, w in [("DA_T010_head", .12), ("DB_T020_head", .12), ("DC_T005_head", .12)]:
        src = {"DA_T010_head": "H2T010", "DB_T020_head": "H2T020", "DC_T005_head": "H2T005"}[tag]
        emit(tag, {"ours": .15, "binary": .30, src: w, "H94": .215, "H95": .215}, TAIL_BEST)
    print("=== DD：尾部投票池扩容（多容量×多种子）===")
    emit("DD0_ref", HEAD, TAIL_BEST)
    emit("DD1_pool_all", HEAD,
         {"ours": .15, "binary": .10, "H94": .28, "H2L15": .07, "H2L11": .07, "H2L9": .07,
          "H2L7": .07, "H2L07s07": .06, "H2L11s24": .06, "H2L15s24": .02, "H95": .12})
    emit("DD2_pool_seed", HEAD,
         {"ours": .15, "binary": .10, "H94": .28, "H2L15": .07, "H2L7": .07,
          "H2L07s07": .07, "H2L11s24": .07, "H2L15s24": .07, "H95": .10})
    emit("DD3_headfocus_tailpool", {"ours": .15, "binary": .30, "H2T010": .12, "H94": .215, "H95": .215},
         {"ours": .15, "binary": .10, "H94": .28, "H2L15": .07, "H2L11": .07, "H2L9": .07,
          "H2L7": .07, "H2L07s07": .06, "H2L11s24": .06, "H2L15s24": .02, "H95": .12})
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
