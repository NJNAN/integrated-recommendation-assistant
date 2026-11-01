#!/usr/bin/env python3
"""
v6 优化：更丰富的特征（复购周期/到期度、国家匹配、热度趋势、篮子共现）+ 多seed LightGBM bagging
流程：随机划分验证 -> 网格搜索融合权重 -> 全量训练 -> 生成 submission
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

TRAIN_PATH = r"D:\edge下载\train (1).csv"
CAND_PATH = r"D:\edge下载\sample_candidates.csv"
OUTPUT_DIR = r"D:\CodeBuddy\recommendation"
MODEL_VERSION = "v6.0_feat_bagging"

SEED = 42
np.random.seed(SEED)
N_SVD = 64
N_NEG = 4


# --------------------------- 通用工具 ---------------------------
def metrics(y, s, groups, k_list=(10, 20)):
    df = pd.DataFrame({"y": y, "s": s, "g": groups})
    gaucs, ndcg_full = [], []
    ndcgs = {k: [] for k in k_list}
    recalls = {k: [] for k in k_list}
    for _, g in df.groupby("g"):
        if g["y"].nunique() < 2:
            continue
        yt = g["y"].values
        ss = g["s"].values
        gaucs.append(roc_auc_score(yt, ss))
        order = np.argsort(-ss)
        ys = yt[order]
        npos = yt.sum()
        disc = 1.0 / np.log2(np.arange(2, len(ys) + 2))
        dcg = np.sum((2 ** ys - 1) * disc)
        ideal = np.sort(yt)[::-1]
        idcg = np.sum((2 ** ideal - 1) * disc)
        ndcg_full.append(dcg / idcg if idcg > 0 else 0)
        for k in k_list:
            top = ys[:k]
            dcgk = np.sum((2 ** top - 1) / np.log2(np.arange(2, len(top) + 2)))
            ik = np.sort(yt)[::-1][:k]
            idcgk = np.sum((2 ** ik - 1) / np.log2(np.arange(2, len(ik) + 2)))
            ndcgs[k].append(dcgk / idcgk if idcgk > 0 else 0)
            recalls[k].append(top.sum() / npos if npos > 0 else 0)
    r = {"GAUC": float(np.mean(gaucs)), "NDCGfull": float(np.mean(ndcg_full))}
    for k in k_list:
        r[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
        r[f"R@{k}"] = float(np.mean(recalls[k]))
    return r


def comp_of(r):
    return 0.4 * r["NDCGfull"] + 0.4 * r["GAUC"] + 0.2 * r["R@20"]


def load_data():
    tr = pd.read_csv(TRAIN_PATH).rename(columns={"CustomerID": "user_id", "StockCode": "item_id"})
    tr = tr.dropna(subset=["user_id", "item_id", "Quantity", "UnitPrice"])
    tr["user_id"] = tr["user_id"].astype(str)
    tr["item_id"] = tr["item_id"].astype(str)
    tr["Country"] = tr["Country"].fillna("NA").astype(str)
    tr["InvoiceDate"] = pd.to_datetime(tr["InvoiceDate"])
    tr["revenue"] = tr["Quantity"] * tr["UnitPrice"]
    cd = pd.read_csv(CAND_PATH)
    cd["user_id"] = cd["user_id"].astype(str)
    cd["item_id"] = cd["item_id"].astype(str)
    return tr, cd


# --------------------------- CF / SVD ---------------------------
def build_cf(train):
    ui = train[train["Quantity"] > 0].groupby(["user_id", "item_id"]).size().reset_index(name="c")
    ue = LabelEncoder().fit(ui["user_id"])
    ie = LabelEncoder().fit(ui["item_id"])
    R = csr_matrix((np.ones(len(ui)), (ue.transform(ui["user_id"]), ie.transform(ui["item_id"]))),
                   shape=(len(ue.classes_), len(ie.classes_)))
    R.data[:] = 1.0
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    itf = svd.fit_transform(R.T)
    usf = R.dot(itf)
    itf = itf / (np.linalg.norm(itf, axis=1, keepdims=True) + 1e-8)
    usf = usf / (np.linalg.norm(usf, axis=1, keepdims=True) + 1e-8)

    C = (R.T @ R).tocoo()
    C.setdiag(0)
    C.eliminate_zeros()
    dg = np.sqrt(np.asarray(R.sum(axis=0)).ravel())
    dg[dg == 0] = 1.0
    S = csr_matrix((C.data / (dg[C.row] * dg[C.col]), (C.row, C.col)), shape=(len(ie.classes_),) * 2)
    itemcf = np.asarray((R @ S).todense(), dtype=np.float32)

    U = (R @ R.T).tocoo()
    U.setdiag(0)
    U.eliminate_zeros()
    ug = np.sqrt(np.asarray(R.sum(axis=1)).ravel())
    ug[ug == 0] = 1.0
    Us = csr_matrix((U.data / (ug[U.row] * ug[U.col]), (U.row, U.col)), shape=(len(ue.classes_),) * 2)
    usercf = np.asarray((Us @ R).todense(), dtype=np.float32)

    uprof = np.zeros((len(ue.classes_), N_SVD), dtype=np.float32)
    ip, ii = R.indptr, R.indices
    for u in range(len(ue.classes_)):
        h = ii[ip[u]:ip[u + 1]]
        if len(h):
            uprof[u] = itf[h].mean(axis=0)

    return {"ue": ue, "ie": ie, "itf": itf, "usf": usf,
            "itemcf": itemcf, "usercf": usercf, "uprof": uprof}


# --------------------------- 特征查找表 ---------------------------
def build_lookups(train, ref_date):
    pos = train[train["Quantity"] > 0]
    # 复购明细
    ui = pos.groupby(["user_id", "item_id"]).agg(
        cnt=("InvoiceNo", "nunique"), qty=("Quantity", "sum"),
        last=("InvoiceDate", "max"), first=("InvoiceDate", "min"),
        ndates=("InvoiceDate", "nunique"),
    ).reset_index()
    ui["interval"] = (ui["last"] - ui["first"]).dt.days / (ui["ndates"] - 1).clip(lower=1)
    ui["days_since_last"] = (ref_date - ui["last"]).dt.days
    ui["due_ratio"] = ui["days_since_last"] / (ui["interval"] + 1.0)
    ui["k"] = ui["user_id"] + "|" + ui["item_id"]

    user_stats = train.groupby("user_id").agg(
        user_total_orders=("InvoiceNo", "nunique"),
        user_unique_items=("item_id", "nunique"),
        user_total_spend=("revenue", "sum"),
        user_avg_price=("UnitPrice", "mean"),
        user_std_price=("UnitPrice", "std"),
        user_avg_qty=("Quantity", "mean"),
    ).reset_index()
    user_stats["user_std_price"] = user_stats["user_std_price"].fillna(0)

    item_stats = train.groupby("item_id").agg(
        item_total_orders=("InvoiceNo", "nunique"),
        item_unique_users=("user_id", "nunique"),
        item_total_qty=("Quantity", "sum"),
        item_avg_price=("UnitPrice", "mean"),
        item_avg_qty=("Quantity", "mean"),
    ).reset_index()
    item_stats["item_pop_rank"] = item_stats["item_total_orders"].rank(method="dense", ascending=False)

    # 近30/90天热度趋势
    for w in (30, 90):
        start = ref_date - pd.Timedelta(days=w)
        recent = train[train["InvoiceDate"] >= start]
        ir = recent.groupby("item_id").agg(**{
            f"item_orders_{w}d": ("InvoiceNo", "nunique"),
            f"item_users_{w}d": ("user_id", "nunique"),
        }).reset_index()
        item_stats = item_stats.merge(ir, on="item_id", how="left")
        ur = recent.groupby("user_id").agg(**{
            f"user_orders_{w}d": ("InvoiceNo", "nunique"),
            f"user_items_{w}d": ("item_id", "nunique"),
        }).reset_index()
        user_stats = user_stats.merge(ur, on="user_id", how="left")
    for c in ["item_orders_30d", "item_users_30d", "item_orders_90d", "item_users_90d"]:
        item_stats[c] = item_stats[c].fillna(0)
    for c in ["user_orders_30d", "user_items_30d", "user_orders_90d", "user_items_90d"]:
        user_stats[c] = user_stats[c].fillna(0)
    item_stats["item_trend_30"] = item_stats["item_orders_30d"] / (item_stats["item_total_orders"] + 1.0)
    user_stats["user_active_30"] = user_stats["user_orders_30d"] / (user_stats["user_total_orders"] + 1.0)

    # 用户最近一次购买
    rec = train.groupby("user_id")["InvoiceDate"].max().reset_index(name="user_last_date")
    user_stats = user_stats.merge(rec, on="user_id", how="left")
    user_stats["user_days_since_last"] = (ref_date - user_stats["user_last_date"]).dt.days

    # 国家
    user_country = train.groupby("user_id")["Country"].agg(lambda x: x.mode().iloc[0]).reset_index(name="user_country")
    user_stats = user_stats.merge(user_country, on="user_id", how="left")
    item_country = pos.groupby("item_id")["Country"].agg(lambda x: x.mode().iloc[0]).reset_index(name="item_top_country")
    item_stats = item_stats.merge(item_country, on="item_id", how="left")
    # 商品在用户所属国家的销量
    ic = pos.groupby(["item_id", "Country"]).size().reset_index(name="ic_cnt")
    ic["k"] = ic["item_id"] + "|" + ic["Country"]

    return {
        "ui": ui,
        "user_stats": user_stats,
        "item_stats": item_stats,
        "ic_map": dict(zip(ic["k"], ic["ic_cnt"])),
        "ui_cnt_map": dict(zip(ui["k"], ui["cnt"])),
    }


def apply_features(pairs, train, lookups, cf):
    df = pairs.copy()
    us = lookups["user_stats"].drop(columns=["user_last_date"])
    it = lookups["item_stats"]
    df = df.merge(us, on="user_id", how="left").merge(it, on="item_id", how="left")

    # CF / SVD
    ue, ie = cf["ue"], cf["ie"]
    umap = {v: k for k, v in enumerate(ue.classes_)}
    imap = {v: k for k, v in enumerate(ie.classes_)}
    uidx = np.array([umap.get(x, -1) for x in df["user_id"]])
    iidx = np.array([imap.get(x, -1) for x in df["item_id"]])
    valid = (uidx >= 0) & (iidx >= 0)
    df["cf_item"] = 0.0
    df["cf_user"] = 0.0
    df["svd_dot"] = 0.0
    df["hist_sim"] = 0.0
    df.loc[valid, "cf_item"] = cf["itemcf"][uidx[valid], iidx[valid]]
    df.loc[valid, "cf_user"] = cf["usercf"][uidx[valid], iidx[valid]]
    df.loc[valid, "svd_dot"] = np.sum(cf["usf"][uidx[valid]] * cf["itf"][iidx[valid]], axis=1)
    df.loc[valid, "hist_sim"] = np.sum(cf["uprof"][uidx[valid]] * cf["itf"][iidx[valid]], axis=1)

    # 复购
    kk = df["user_id"] + "|" + df["item_id"]
    df["rep_cnt"] = kk.map(lookups["ui_cnt_map"]).fillna(0)
    ui = lookups["ui"].set_index("k")
    for col, name in [("qty", "rep_qty"), ("ndates", "rep_ndates"), ("interval", "rep_interval"),
                      ("days_since_last", "rep_days_since_last"), ("due_ratio", "rep_due_ratio")]:
        df[name] = kk.map(ui[col]).fillna(-1)
    df["rep_share_user"] = df["rep_cnt"] / (df["user_total_orders"].fillna(0) + 1.0)
    df["rep_share_item"] = df["rep_cnt"] / (df["item_unique_users"].fillna(0) + 1.0)

    # 国家
    df["country_match"] = (df["user_country"] == df["item_top_country"]).astype(int)
    df["item_in_user_country"] = (df["item_id"] + "|" + df["user_country"].fillna("NA")).map(
        lookups["ic_map"]).fillna(0)

    # 价格
    df["price_diff"] = df["item_avg_price"] - df["user_avg_price"]
    df["price_ratio"] = df["item_avg_price"] / (df["user_avg_price"] + 1e-8)
    df["price_z"] = (df["item_avg_price"] - df["user_avg_price"]) / (df["user_std_price"] + 1e-8)

    # 热度交互
    df["pop_x_user"] = df["item_total_orders"] * np.log1p(df["user_total_orders"])
    df["user_item_rank_ratio"] = df["user_total_orders"] / (df["item_pop_rank"] + 1.0)

    num_cols = [c for c in df.columns if c not in ("user_id", "item_id", "label", "user_country", "item_top_country")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df


FEATS = [
    "cf_item", "cf_user", "svd_dot", "hist_sim",
    "rep_cnt", "rep_qty", "rep_ndates", "rep_interval", "rep_days_since_last", "rep_due_ratio",
    "rep_share_user", "rep_share_item",
    "country_match", "item_in_user_country",
    "price_diff", "price_ratio", "price_z",
    "user_total_orders", "user_unique_items", "user_total_spend", "user_avg_price",
    "user_std_price", "user_avg_qty", "user_orders_30d", "user_items_30d",
    "user_orders_90d", "user_items_90d", "user_active_30", "user_days_since_last",
    "item_total_orders", "item_unique_users", "item_total_qty", "item_avg_price",
    "item_avg_qty", "item_pop_rank", "item_orders_30d", "item_users_30d",
    "item_orders_90d", "item_users_90d", "item_trend_30",
    "pop_x_user", "user_item_rank_ratio",
]


def build_training_samples(train, n_neg, rng):
    pos = train[train["Quantity"] > 0][["user_id", "item_id"]].drop_duplicates()
    pos["label"] = 1
    user_items = train.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = np.array(list(train["item_id"].unique()))
    pop = train.groupby("item_id")["InvoiceNo"].nunique()
    w_all = np.array([pop.get(it, 1.0) for it in all_items], dtype=np.float64)
    w_all = w_all / w_all.sum()
    negs = []
    for u, b in user_items.items():
        mask = ~np.isin(all_items, list(b))
        av = all_items[mask]
        w = w_all[mask]
        if len(av) == 0 or w.sum() == 0:
            continue
        w = w / w.sum()
        ns = min(n_neg * len(b), len(av))
        sampled = rng.choice(av, size=ns, replace=False, p=w)
        for it in sampled:
            negs.append((u, it, 0))
    neg = pd.DataFrame(negs, columns=["user_id", "item_id", "label"])
    return pd.concat([pos, neg], ignore_index=True)


def rank_norm(df, col):
    r = df.groupby("user_id")[col].rank(method="average")
    n = df.groupby("user_id")[col].transform("size")
    return (r / n).values


def train_bagged(X, y, rounds, seeds=(42, 2024, 7), params=None):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    base = {"objective": "binary", "metric": "auc", "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_data_in_leaf": 80, "scale_pos_weight": spw, "verbose": -1, "num_threads": 0}
    if params:
        base.update(params)
    preds = np.zeros(len(X))
    for s in seeds:
        m = lgb.train({**base, "seed": s}, lgb.Dataset(X, label=y), num_boost_round=rounds)
        preds += m.predict(X)
    return preds / len(seeds)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train, cand = load_data()
    cand_pairs = cand[["user_id", "item_id"]].drop_duplicates().copy()
    print(f"train={len(train):,} cand={len(cand_pairs):,}", flush=True)

    # ---------- 随机划分验证 ----------
    rng = np.random.RandomState(SEED)
    mask = rng.rand(len(train)) < 0.8
    tp, vp = train[mask].copy(), train[~mask].copy()
    ref_val = tp["InvoiceDate"].max() + pd.Timedelta(days=1)

    cf = build_cf(tp)
    lk = build_lookups(tp, ref_val)

    pos_keys = set(zip(*(vp[["user_id", "item_id"]].drop_duplicates().values.T)))
    val_df = apply_features(cand_pairs, tp, lk, cf)
    val_df["label"] = [1 if (u, i) in pos_keys else 0
                       for u, i in zip(val_df["user_id"], val_df["item_id"])]
    y_val = val_df["label"].values
    groups = val_df["user_id"].values
    print(f"val pos rate={y_val.mean():.4f} pos={y_val.sum():,}", flush=True)

    tr_samples = build_training_samples(tp, N_NEG, np.random.RandomState(SEED))
    tr_df = apply_features(tr_samples, tp, lk, cf)

    # 单模型（早停）确定轮数
    spw = (tr_df["label"] == 0).sum() / max((tr_df["label"] == 1).sum(), 1)
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "num_leaves": 63, "learning_rate": 0.05,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
         "min_data_in_leaf": 80, "scale_pos_weight": spw, "verbose": -1, "seed": SEED},
        lgb.Dataset(tr_df[FEATS], label=tr_df["label"]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val_df[FEATS], label=y_val)],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)],
    )
    best_rounds = model.best_iteration
    print(f"best_rounds={best_rounds}", flush=True)
    pred_single = model.predict(val_df[FEATS], num_iteration=best_rounds)

    tmp = val_df[["user_id"]].copy()
    tmp["gbdt"] = pred_single
    gb_rank = rank_norm(tmp, "gbdt")
    rep_rank = rank_norm(val_df, "rep_cnt")
    svd_rank = rank_norm(val_df, "svd_dot")
    cf_rank = rank_norm(val_df, "cf_item")
    pop_rank = rank_norm(val_df, "item_total_orders")

    for name, s in [("GBDT", pred_single), ("rank_blend_v5", 0.35 * gb_rank + 0.35 * rep_rank +
                                            0.1 * svd_rank + 0.1 * cf_rank + 0.1 * pop_rank)]:
        r = metrics(y_val, s, groups)
        print(f"{name:16s} GAUC={r['GAUC']:.4f} NDCGfull={r['NDCGfull']:.4f} "
              f"NDCG@20={r['NDCG@20']:.4f} R@20={r['R@20']:.4f} comp={comp_of(r):.4f}", flush=True)

    # 网格搜索权重
    best = None
    for w_gb in [0.2, 0.3, 0.35, 0.45, 0.55]:
        for w_rep in [0.2, 0.3, 0.35, 0.45]:
            for w_svd in [0.05, 0.1, 0.15]:
                for w_cf in [0.0, 0.05, 0.1]:
                    w_pop = 1.0 - w_gb - w_rep - w_svd - w_cf
                    if w_pop < -1e-9 or w_pop > 0.2:
                        continue
                    w_pop = max(w_pop, 0.0)
                    s = (w_gb * gb_rank + w_rep * rep_rank + w_svd * svd_rank +
                         w_cf * cf_rank + w_pop * pop_rank)
                    r = metrics(y_val, s, groups)
                    c = comp_of(r)
                    if best is None or c > best[0]:
                        best = (c, (w_gb, w_rep, w_svd, w_cf, round(w_pop, 2)), r)
    c, w, r = best
    print(f"BEST weights gb/rep/svd/cf/pop={w} comp={c:.4f} GAUC={r['GAUC']:.4f} "
          f"NDCGfull={r['NDCGfull']:.4f} R@20={r['R@20']:.4f}", flush=True)

    # ---------- 全量训练 ----------
    print("Training final on full data...", flush=True)
    ref_full = train["InvoiceDate"].max() + pd.Timedelta(days=1)
    cf_f = build_cf(train)
    lk_f = build_lookups(train, ref_full)
    full_samples = build_training_samples(train, N_NEG, np.random.RandomState(SEED))
    full_df = apply_features(full_samples, train, lk_f, cf_f)
    cand_df = apply_features(cand_pairs, train, lk_f, cf_f)

    spw_f = (full_df["label"] == 0).sum() / max((full_df["label"] == 1).sum(), 1)
    base = {"objective": "binary", "metric": "auc", "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_data_in_leaf": 80, "scale_pos_weight": spw_f, "verbose": -1, "num_threads": 0}
    cand_pred = np.zeros(len(cand_df))
    for s in (42, 2024, 7):
        m = lgb.train({**base, "seed": s}, lgb.Dataset(full_df[FEATS], label=full_df["label"]),
                      num_boost_round=best_rounds)
        cand_pred += m.predict(cand_df[FEATS])
    cand_pred /= 3.0
    cand_df["gbdt"] = cand_pred

    gb = rank_norm(cand_df, "gbdt")
    rep = rank_norm(cand_df, "rep_cnt")
    svd = rank_norm(cand_df, "svd_dot")
    cfr = rank_norm(cand_df, "cf_item")
    pop = rank_norm(cand_df, "item_total_orders")
    score = w[0] * gb + w[1] * rep + w[2] * svd + w[3] * cfr + w[4] * pop

    submission = pd.DataFrame({"user_id": cand_df["user_id"].values,
                               "item_id": cand_df["item_id"].values, "score": score})
    assert len(submission) == len(cand_pairs)
    assert submission["score"].isna().sum() == 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"submission_{MODEL_VERSION}_{ts}.csv")
    submission.to_csv(out_file, index=False, float_format="%.6f")
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False, float_format="%.6f")
    print(f"Saved: {out_file}", flush=True)
    print(submission["score"].describe())
    print(submission.head(10))


if __name__ == "__main__":
    main()
