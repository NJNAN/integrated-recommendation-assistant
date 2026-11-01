#!/usr/bin/env python3
"""
最终提交 v10：v9 EASE 多视图 + LightGBM LambdaRank（最近30天未来购买标签）。
训练时只使用 cutoff 之前的数据构建特征；推理时使用全量训练数据特征。
融合：90% ranker 排名 + 10% v9 排名。
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

import optimize_v6 as o

warnings.filterwarnings("ignore")

TRAIN_PATH = r"D:\edge下载\train (1).csv"
CAND_PATH = r"D:\edge下载\sample_candidates.csv"
OUTPUT_DIR = r"D:\CodeBuddy\recommendation"
MODEL_VERSION = "v10.0_ease_lambdarank"

TAU = 1000.0
LAM_BIN = 200.0
LAM_DAY = 200.0
LAM_REC = 120.0
PRED_WINDOW_DAYS = 30
RANKER_W = 0.90


def ease_B(G, lam):
    G = G.copy()
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.0)
    return B


def gram(X):
    return (X.T @ X).toarray().astype(np.float64)


def add_ease_scores(df, train_for_ease, cand_pairs):
    pos = train_for_ease[train_for_ease["Quantity"] > 0].copy()
    ref = pos["InvoiceDate"].max() + pd.Timedelta(days=1)
    pos["days"] = (ref - pos["InvoiceDate"]).dt.days
    pos["date"] = pos["InvoiceDate"].dt.normalize()
    ie = LabelEncoder().fit(pos["item_id"])
    imap = {v: k for k, v in enumerate(ie.classes_)}
    ni = len(ie.classes_)
    pos["i"] = ie.transform(pos["item_id"])
    ue = LabelEncoder().fit(pos["user_id"])
    umap = {v: k for k, v in enumerate(ue.classes_)}
    nu = len(ue.classes_)
    pos["u"] = ue.transform(pos["user_id"])

    g = pos.groupby(["u", "i"]).agg(c=("i", "size"), d=("days", "min")).reset_index()
    ui, ii, n = g["u"].values, g["i"].values, len(g)
    Xb = csr_matrix((np.ones(n), (ui, ii)), shape=(nu, ni)).tocsr()
    Xr = csr_matrix((np.exp(-g["d"].values / TAU) * np.log1p(g["c"].values), (ui, ii)),
                    shape=(nu, ni)).tocsr()

    daykey = pos["u"].astype(str) + "@" + pos["date"].astype(str)
    be = LabelEncoder().fit(daykey)
    di = pd.DataFrame({"b": be.transform(daykey), "i": pos["i"].values}).drop_duplicates()
    Xday = csr_matrix((np.ones(len(di)), (di["b"].values, di["i"].values)),
                      shape=(len(be.classes_), ni)).tocsr()

    uidx = np.array([umap.get(x, -1) for x in cand_pairs["user_id"]])
    iidx = np.array([imap.get(x, -1) for x in cand_pairs["item_id"]])
    valid = (uidx >= 0) & (iidx >= 0)

    def score(P, X, lam):
        S = P @ ease_B(gram(X), lam)
        s = np.full(len(cand_pairs), np.nan)
        s[valid] = S[uidx[valid], iidx[valid]]
        return np.where(np.isnan(s), -1e18, s)

    return pd.DataFrame({
        "ease_bin": score(Xr, Xb, LAM_BIN),
        "ease_day": score(Xr, Xday, LAM_DAY),
        "ease_rec": score(Xr, Xr, LAM_REC),
    })


def build_feature_df(train, cand_pairs, cutoff_days=None):
    """cutoff_days=None 表示使用全量数据；否则只用 cutoff 之前的数据构建特征。"""
    if cutoff_days is not None:
        cutoff = train["InvoiceDate"].max() - pd.Timedelta(days=cutoff_days)
        tp = train[train["InvoiceDate"] < cutoff].copy()
        vp = train[(train["InvoiceDate"] >= cutoff) & (train["Quantity"] > 0)].copy()
    else:
        tp = train.copy()
        vp = None

    ref = tp["InvoiceDate"].max() + pd.Timedelta(days=1)
    cf = o.build_cf(tp)
    lk = o.build_lookups(tp, ref)
    base = o.apply_features(cand_pairs, tp, lk, cf)
    if vp is not None:
        pos_keys = set(zip(*(vp[["user_id", "item_id"]].drop_duplicates().values.T)))
        base["label"] = [1 if (u, i) in pos_keys else 0
                         for u, i in zip(base["user_id"], base["item_id"])]
    else:
        base["label"] = np.nan

    ease = add_ease_scores(base, tp, cand_pairs)
    base = pd.concat([base.reset_index(drop=True), ease.reset_index(drop=True)], axis=1)
    for c in ["ease_bin", "ease_day", "ease_rec", "cf_item", "cf_user", "svd_dot", "hist_sim",
              "rep_cnt", "item_total_orders"]:
        base[f"{c}_rank"] = o.rank_norm(base, c)
    base["ease_rank_ens"] = base[["ease_bin_rank", "ease_day_rank", "ease_rec_rank"]].mean(axis=1)
    return base


def train_ranker(df, feats, seed=42):
    """用用户级划分早停，返回最佳迭代轮数。"""
    users = df["user_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n_tr = int(0.8 * len(users))
    tr_df = df[df["user_id"].isin(set(users[:n_tr]))].copy()
    va_df = df[df["user_id"].isin(set(users[n_tr:]))].copy()

    Xtr, ytr = tr_df[feats].astype(float), tr_df["label"].values
    Xva, yva = va_df[feats].astype(float), va_df["label"].values
    gtr = tr_df.groupby("user_id").size().values
    gva = va_df.groupby("user_id").size().values

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20, 1000],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "verbose": -1,
        "seed": seed,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
    }
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr)
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dtrain, dval],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    print(f"ranker best_iter={model.best_iteration}", flush=True)
    return model.best_iteration


def v9_score(train, cand_pairs):
    """复用 final_v9 逻辑：全量数据、lambda 放大 1.25 倍、等权排名融合。"""
    pos = train[train["Quantity"] > 0].copy()
    ref = pos["InvoiceDate"].max() + pd.Timedelta(days=1)
    pos["days"] = (ref - pos["InvoiceDate"]).dt.days
    pos["date"] = pos["InvoiceDate"].dt.normalize()
    ie = LabelEncoder().fit(pos["item_id"])
    imap = {v: k for k, v in enumerate(ie.classes_)}
    ni = len(ie.classes_)
    pos["i"] = ie.transform(pos["item_id"])
    ue = LabelEncoder().fit(pos["user_id"])
    umap = {v: k for k, v in enumerate(ue.classes_)}
    nu = len(ue.classes_)
    pos["u"] = ue.transform(pos["user_id"])

    g = pos.groupby(["u", "i"]).agg(c=("i", "size"), d=("days", "min")).reset_index()
    ui, ii, n = g["u"].values, g["i"].values, len(g)
    Xb = csr_matrix((np.ones(n), (ui, ii)), shape=(nu, ni)).tocsr()
    Xr = csr_matrix((np.exp(-g["d"].values / TAU) * np.log1p(g["c"].values), (ui, ii)),
                    shape=(nu, ni)).tocsr()
    daykey = pos["u"].astype(str) + "@" + pos["date"].astype(str)
    be = LabelEncoder().fit(daykey)
    di = pd.DataFrame({"b": be.transform(daykey), "i": pos["i"].values}).drop_duplicates()
    Xday = csr_matrix((np.ones(len(di)), (di["b"].values, di["i"].values)),
                      shape=(len(be.classes_), ni)).tocsr()

    uidx = np.array([umap.get(x, -1) for x in cand_pairs["user_id"]])
    iidx = np.array([imap.get(x, -1) for x in cand_pairs["item_id"]])
    valid = (uidx >= 0) & (iidx >= 0)
    out = pd.DataFrame({"user_id": cand_pairs["user_id"].values})
    ranks = []
    for P, X, lam in [(Xr, Xb, 250.0), (Xr, Xday, 250.0), (Xr, Xr, 150.0)]:
        S = P @ ease_B(gram(X), lam)
        s = np.full(len(cand_pairs), np.nan)
        s[valid] = S[uidx[valid], iidx[valid]]
        out["v"] = np.where(np.isnan(s), -1e18, s)
        ranks.append((out.groupby("user_id")["v"].rank(method="average") /
                      out.groupby("user_id")["v"].transform("size")).values)
    return np.mean(ranks, axis=0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train, cd = o.load_data()
    cand_pairs = cd[["user_id", "item_id"]].drop_duplicates().copy()
    print(f"train={len(train):,} cand={len(cand_pairs):,}", flush=True)

    base_feats = o.FEATS + ["ease_bin", "ease_day", "ease_rec",
                            "ease_bin_rank", "ease_day_rank", "ease_rec_rank", "ease_rank_ens",
                            "cf_item_rank", "cf_user_rank", "svd_dot_rank", "hist_sim_rank",
                            "rep_cnt_rank", "item_total_orders_rank"]

    # 1) 用 cutoff 之前的数据构建训练集（最近30天作标签），用户级划分早停
    print("Building train features (before cutoff)...", flush=True)
    df_train = build_feature_df(train, cand_pairs, cutoff_days=PRED_WINDOW_DAYS)
    print(f"train labels pos={int(df_train['label'].sum())} rate={df_train['label'].mean():.3f}", flush=True)
    best_iter = train_ranker(df_train, base_feats, seed=42)

    # 2) 用全量用户（cutoff 之前特征）重新训练到最佳轮数
    print("Retraining ranker on all users...", flush=True)
    Xtr_all = df_train[base_feats].astype(float)
    ytr_all = df_train["label"].values
    gtr_all = df_train.groupby("user_id").size().values
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10, 20, 1000],
        "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_data_in_leaf": 100,
        "verbose": -1, "seed": 42, "lambda_l1": 1.0, "lambda_l2": 10.0,
    }
    model = lgb.train(params, lgb.Dataset(Xtr_all, label=ytr_all, group=gtr_all),
                      num_boost_round=best_iter)

    # 3) 用全量数据构建候选特征并预测
    print("Building full features and scoring...", flush=True)
    df_full = build_feature_df(train, cand_pairs, cutoff_days=None)
    ranker_score = model.predict(df_full[base_feats].astype(float))

    # 4) v9 全量分数
    print("Building v9 score...", flush=True)
    v9_rank = v9_score(train, cand_pairs)

    # 4) 排名融合
    tmp = pd.DataFrame({"user_id": cand_pairs["user_id"].values, "v": ranker_score})
    ranker_rank = (tmp.groupby("user_id")["v"].rank(method="average") /
                   tmp.groupby("user_id")["v"].transform("size")).values
    final_score = RANKER_W * ranker_rank + (1 - RANKER_W) * v9_rank

    submission = pd.DataFrame({"user_id": cand_pairs["user_id"].values,
                               "item_id": cand_pairs["item_id"].values, "score": final_score})
    assert len(submission) == len(cand_pairs)
    assert submission["score"].isna().sum() == 0
    assert not submission.duplicated(["user_id", "item_id"]).any()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"submission_{MODEL_VERSION}_{ts}.csv")
    submission.to_csv(out_file, index=False, float_format="%.6f")
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False, float_format="%.6f")
    print(f"Saved: {out_file}", flush=True)
    print(submission["score"].describe())


if __name__ == "__main__":
    main()
