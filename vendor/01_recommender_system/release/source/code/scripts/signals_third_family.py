"""A：第三家族模型的信号层——所有人没用过的三个特征族。

为什么要有这个
--------------
目前所有可用伙伴（第9组 v10/v61/v63/v65/v67）**共用同一套 49 特征**，
彼此 Spearman 0.88~0.98，与本组冠军也都在 0.75~0.78。
它们本质上是同一个家族，所以融合增益到 0.8005 后开始收敛。

本模块只造**三个尚未被任何一方使用的信号**：

1. 文本语义相似度
   商品描述（Description）的字符 n-gram TF-IDF 向量余弦。
   与"21 类粗分类"（v52 试过、结果负）不同，这是细粒度语义：
   "RED WALL CLOCK" 与 "RETROSPOT WALL CLOCK" 高度相似。

2. 发货单级关联规则
   同一张发票内商品的共现提升度 lift = P(i,j)/(P(i)P(j))，带收缩估计。
   与 EASE 不同：EASE 是全局线性重构（稠密逆矩阵），这里是成对统计量，
   对长尾商品的局部搭配更敏感。

3. StockCode 商品族
   同一字母前缀（如 "85123A" / "85099B"）代表同类商品族，
   用户买过某族的商品后买同族其他商品的倾向。

时间安全性
----------
每个窗口的特征**只用到该窗口 cutoff 之前的交易**。文本向量虽由历史描述构成，
但描述是商品静态属性，且同样取自 cutoff 之前的历史，故不引入未来信息。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]


def load_clean() -> pd.DataFrame:
    tr = pd.read_csv(ROOT / "data/train.csv")
    tr = tr.dropna(subset=["CustomerID", "StockCode", "Quantity", "UnitPrice"])
    tr = tr.rename(columns={"CustomerID": "user_id", "StockCode": "item_id",
                            "InvoiceDate": "transaction_time", "InvoiceNo": "invoice_no",
                            "Description": "description"})
    tr["user_id"] = tr["user_id"].astype(int).astype(str)
    tr["item_id"] = tr["item_id"].astype(str)
    tr["invoice_no"] = tr["invoice_no"].astype(str)
    tr["transaction_time"] = pd.to_datetime(tr["transaction_time"])
    return tr


def _family(item: str) -> str:
    letters = "".join(ch for ch in str(item) if ch.isalpha())
    return letters[:1].upper() if letters else "?"


def build_signals(rows: pd.DataFrame, history: pd.DataFrame, *, top_bought: int = 20) -> pd.DataFrame:
    """为 rows（user_id/item_id，无标签）计算新信号特征。

    rows     : 候选对，仅 ID 列。
    history  : 特征期交易（transaction_time < cutoff），已断言其上界。
    """
    assert set(rows.columns) == {"user_id", "item_id"}, "只接受 ID 列，防止标签渗入"
    out = rows.astype(str).reset_index(drop=True).copy()
    hist = history.copy()

    # ---------- 1. 文本语义相似度 ----------
    desc = hist.groupby("item_id", sort=False).description.last().astype(str)
    vocab = desc.index.tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    X = normalize(vec.fit_transform(desc.fillna("").values))
    item_pos = {it: i for i, it in enumerate(vocab)}

    # 用户最近购买的 top_bought 个商品（正数量）
    bought = hist.loc[hist.Quantity > 0, ["user_id", "item_id", "transaction_time"]]
    recent = (bought.sort_values("transaction_time")
              .groupby("user_id", sort=False).item_id.apply(lambda s: list(dict.fromkeys(s))[-top_bought:]))

    # ---------- 2. 发货单级共现 ----------
    # 只需要 (候选商品, 用户曾买商品) 的 lift，因此先按候选商品集合过滤右表，
    # 避免 40 万行发票边自连接产生的数千万对。
    edges = hist.loc[hist.Quantity > 0, ["invoice_no", "item_id"]].drop_duplicates()
    n_inv = edges.invoice_no.nunique()
    ic = edges.groupby("item_id", sort=False).invoice_no.nunique()
    cand_items = set(out.item_id.unique())
    pair = (edges[edges.item_id.isin(cand_items)]
            .merge(edges, on="invoice_no", suffixes=("_x", "_y")))
    pair = pair[pair.item_id_x != pair.item_id_y][["item_id_x", "item_id_y"]]
    co = pair.groupby(["item_id_x", "item_id_y"], sort=False).size().rename("c").reset_index()
    co["lift"] = (co.c / n_inv) / (
        (co.item_id_x.map(ic) / n_inv) * (co.item_id_y.map(ic) / n_inv) + 1e-9)
    co["lift"] = co.lift * co.c / (co.c + 20.0)          # 收缩：少样本向 0 收缩
    lut_fwd = co.set_index(["item_id_x", "item_id_y"]).lift.to_dict()
    lut_rev = co.set_index(["item_id_y", "item_id_x"]).lift.to_dict()
    lut_co_f = co.set_index(["item_id_x", "item_id_y"]).c.to_dict()
    lut_co_r = co.set_index(["item_id_y", "item_id_x"]).c.to_dict()

    # ---------- 3. 商品族 ----------
    fam = {it: _family(it) for it in hist.item_id.unique()}
    pop = hist.loc[hist.Quantity > 0].groupby("item_id", sort=False).size()
    upop = hist.loc[hist.Quantity > 0].groupby("user_id", sort=False).size()

    rows_out = []
    for uid, grp in out.groupby("user_id", sort=False):
        bl = recent.get(uid, [])
        bp = [item_pos[i] for i in bl if i in item_pos]
        bset = set(bl)
        bfam = pd.Series([fam.get(i, "?") for i in bl]).value_counts().to_dict() if bl else {}
        for it in grp.item_id:
            rec = {}
            p = item_pos.get(it)
            if bp and p is not None:
                sims = X[p].dot(X[bp].T).toarray().ravel()
                rec["a_text_sim_max"] = float(sims.max())
                rec["a_text_sim_mean"] = float(sims.mean())
                rec["a_text_sim_top3"] = float(np.sort(sims)[-3:].mean())
                rec["a_text_sim_cnt50"] = float((sims >= 0.5).sum())
            else:
                rec.update({"a_text_sim_max": 0.0, "a_text_sim_mean": 0.0,
                            "a_text_sim_top3": 0.0, "a_text_sim_cnt50": 0.0})
            lf = [lut_fwd.get((i, it), lut_rev.get((it, i), 0.0)) for i in bl] or [0.0]
            cf = [lut_co_f.get((i, it), lut_co_r.get((it, i), 0.0)) for i in bl] or [0.0]
            rec["a_lift_max"] = float(np.max(lf))
            rec["a_lift_mean"] = float(np.mean(lf))
            rec["a_cooc_max"] = float(np.max(cf))
            rec["a_cooc_mean"] = float(np.mean(cf))
            f = fam.get(it, "?")
            rec["a_family_seen"] = float(bfam.get(f, 0))
            rec["a_family_rate"] = float(bfam.get(f, 0) / max(len(bl), 1))
            rec["a_item_pop"] = float(np.log1p(pop.get(it, 0)))
            rec["a_user_pop"] = float(np.log1p(upop.get(uid, 0)))
            rec["a_in_bought"] = float(it in bset)
            rec["_u"] = uid
            rec["_i"] = it
            rows_out.append(rec)
    sig = pd.DataFrame(rows_out)
    sig = sig.set_index(["_u", "_i"]).reindex(
        pd.MultiIndex.from_arrays([out.user_id, out.item_id])).reset_index(drop=True)
    sig = sig.astype(np.float32).fillna(0.0)
    return sig


SIGNAL_COLS = ["a_text_sim_max", "a_text_sim_mean", "a_text_sim_top3", "a_text_sim_cnt50",
               "a_lift_max", "a_lift_mean", "a_cooc_max", "a_cooc_mean",
               "a_family_seen", "a_family_rate", "a_item_pop", "a_user_pop", "a_in_bought"]
