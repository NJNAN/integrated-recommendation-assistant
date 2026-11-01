"""O2 / H2：把「我们的 73 特征」按窗口重建，与他们的特征、新信号合体。

为什么现在才做
--------------
我们的特征表（production_v15_features.pkl）用的是**11 月截止点**，含 10 月购买。
直接拿它配 6/7/8 月的标签会造成标签期污染（v29 的失败模式）。
必须先证明能按任意 cutoff 重建 —— 已验证：
    cutoff 2010-06-01 -> prior_bought 正例率 0.0751（与该时点一致率 100%）
    cutoff 2010-09-01 -> 0.1063
    cutoff 2010-11-01 -> 0.1511
重建路径：attach_competition_state(engine, history, candidate_frame) + complete_feature_frame。

两个模型
--------
O2 = 我们的 73 特征 + 新信号(A2b)              -> 更强的「本组侧」
H2 = 他们的 49 特征 + 我们的 73 + 新信号(A2b)   -> 全部信息合体

参考性能（单模型线上分）
    我们的冠军 0.7871 | 他们 v10 0.7912 | A2b(他们49+新信号) 0.7998
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "0.7912的模型/recommendation"
CACHE = ROOT / "outputs/competition_v65_capacity/cache"
OUT = ROOT / "outputs/competition_v94_O2_H2"
OUT.mkdir(parents=True, exist_ok=True)
TOP_BOUGHT = 50
ROUNDS = 120
SEED = 42

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))
from baseline.competition import COMPETITION_FEATURES, attach_competition_state   # noqa: E402
from baseline.seasonal_features import SEASONAL_FEATURES, build_seasonal_state    # noqa: E402
from baseline.features_v15 import INTERACTION_FEATURES_V15                        # noqa: E402
from baseline.engine import BaselineEngine                                        # noqa: E402
from scripts.make_submission_v15 import complete_feature_frame                    # noqa: E402
from scripts.train_competition import load_training_data                          # noqa: E402
from scripts.signals_third_family import build_signals, SIGNAL_COLS               # noqa: E402
import optimize_v6 as o                                                           # noqa: E402
o.TRAIN_PATH = str(ROOT / "data/train.csv")
o.CAND_PATH = str(ROOT / "data/sample_candidates.csv")

OUR_FEATS = [*COMPETITION_FEATURES, *SEASONAL_FEATURES, *INTERACTION_FEATURES_V15]
WINDOWS = [("2010-06", "2010-07-01 16:26:00"), ("2010-07", "2010-08-01 16:26:00"),
           ("2010-08", "2010-09-01 16:26:00"), ("2010-09", "2010-10-01 16:26:00"),
           ("2010-10", "2010-10-31 16:26:00")]
ENGINE_DIR = ROOT / "models/competition_v11_season"


def our_features(cands: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """按给定历史重建我们的 73 特征（只接受候选 ID，标签不参与）。"""
    assert hist.transaction_time.max() < pd.Timestamp("2010-11-01")
    eng = BaselineEngine.load(ENGINE_DIR)
    attach_competition_state(eng, hist, candidate_frame=cands)
    full, _ = complete_feature_frame(cands.copy(), eng, build_seasonal_state(hist))
    return full


def urank(v: np.ndarray, uid: np.ndarray) -> np.ndarray:
    s, g = pd.Series(v), pd.Series(uid)
    return (s.groupby(g).rank(method="average") / s.groupby(g).transform("size")).to_numpy()


def main() -> None:
    t0 = time.time()
    clean, _ = load_training_data(ROOT / "data/train.csv")
    assert clean.transaction_time.max() < pd.Timestamp("2010-11-01")
    cand_pairs = pd.read_csv(ROOT / "data/sample_candidates.csv",
                             dtype={"user_id": str, "item_id": str})
    uid = cand_pairs.user_id.to_numpy()

    # 校验重建保真度：11 月重建应复现 pkl
    ref = pd.read_pickle(ROOT / "outputs/competition_v38_categorical/production_v15_features.pkl")
    nov_ours = our_features(cand_pairs, clean)
    chk = [c for c in OUR_FEATS if c in ref.columns and c in nov_ours.columns]
    diff = float(np.abs(nov_ours[chk].astype(float).to_numpy()
                        - ref[chk].astype(float).to_numpy()).max())
    print(f"重建保真度：{len(chk)} 列，与 pkl 最大绝对差 {diff:.3e}  ({time.time()-t0:.0f}s)", flush=True)

    ours_blocks, sig_blocks = [], []
    for name, T in WINDOWS:
        cutoff = pd.Timestamp(T) - pd.Timedelta(days=30)
        hist = clean[clean.transaction_time < cutoff]
        of = our_features(cand_pairs, hist)[OUR_FEATS]
        ours_blocks.append(of)
        sig_blocks.append(build_signals(cand_pairs.copy(), hist, top_bought=TOP_BOUGHT))
        print(f"  {name}: 我方特征 + 新信号完成 ({time.time()-t0:.0f}s)", flush=True)
    sig_nov = build_signals(cand_pairs.copy(), clean, top_bought=TOP_BOUGHT)
    nov_ours = nov_ours[OUR_FEATS]
    print(f"  11月完成 ({time.time()-t0:.0f}s)", flush=True)

    OUR_TR = pd.concat(ours_blocks, ignore_index=True)
    SIG_TR = pd.concat(sig_blocks, ignore_index=True)
    OUR_TR.to_pickle(OUT / "our_feats_train.pkl"); SIG_TR.to_pickle(OUT / "signals_train.pkl")
    nov_ours.to_pickle(OUT / "our_feats_nov.pkl"); sig_nov.to_pickle(OUT / "signals_nov.pkl")

    tr = pd.read_pickle(CACHE / "train_win5.pkl")
    full = pd.read_pickle(CACHE / "full.pkl")
    base_feats = o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                            "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                            "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                            "rep_cnt_rank", "item_total_orders_rank"]
    y = tr["label"].values
    grp = tr.groupby(["_window", "user_id"], sort=False).size().values
    PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
              "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 5, "min_data_in_leaf": 100,
              "verbose": -1, "seed": SEED, "lambda_l1": 1.0, "lambda_l2": 10.0}

    meta = {"outputs": [], "n_our_feats": len(OUR_FEATS)}
    for tag, use_theirs in [("O2", False), ("H2", True)]:
        parts_tr = [OUR_TR.astype(float), SIG_TR[SIGNAL_COLS]]
        parts_no = [nov_ours.astype(float), sig_nov[SIGNAL_COLS]]
        if use_theirs:
            parts_tr = [tr[base_feats].astype(float)] + parts_tr
            parts_no = [full[base_feats].astype(float)] + parts_no
        Xtr = pd.concat(parts_tr, axis=1)
        Xfu = pd.concat(parts_no, axis=1)
        feats = list(Xtr.columns)
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, label=y, group=grp), num_boost_round=ROUNDS)
        rk = urank(m.predict(Xfu), uid)
        f = OUT / f"submission_v94_{tag}.csv"
        pd.DataFrame({"user_id": cand_pairs.user_id.to_numpy(),
                      "item_id": cand_pairs.item_id.to_numpy(), "score": rk}).to_csv(
            f, index=False, float_format="%.6f")
        pd.Series(m.feature_importance("gain"), index=feats).sort_values(
            ascending=False).to_csv(OUT / f"importance_{tag}.csv")
        meta["outputs"].append({"file": f.name, "n_features": len(feats)})
        print(f"  {tag} 完成: {len(feats)} 特征 ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
