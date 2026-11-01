"""v20 共购/购物篮特征：候选商品 vs 用户最近 N 笔订单的匹配。

目标
----
v15 在 8 桶拆解中，"复购 × 低活跃 × 热品"最近月 (2010-10) AUC 仅 0.62，
"首购 × 低活跃 × 热品" (n=33K) AUC 仅 0.73。这两个群体是 v20 的核心目标。

v15 现有特征回答了"用户活跃吗"和"商品热不热"，但没回答"这个商品是否
匹配用户最近的购物篮"。这是 v20 的杠杆点。

3 个新特征（每个都做 max 和 mean，共 6 列）
-------------------------------------------
1. `pair_basket_cooccurrence_{max,mean}_3`：
   用户最近 3 个 invoice 的所有 item 与候选 item 的 P(b|a) 共购分数
   (P(b|a) = P(user 同时有 a 和 b) / P(user 有 a))。
2. `pair_basket_price_gap_{max,mean}_3`：
   |候选均价 - 该 invoice 均价| / 用户均价 std。值越小说明价格档次匹配。
3. `pair_basket_category_overlap_{max,mean}_3`：
   候选 item 的描述关键词集合与该 invoice 内所有 item 的关键词集合
   Jaccard。值越大说明用户最近买的东西和候选属于同主题。

防泄漏
------
- 共现矩阵与用户最近 N 单均从截止时间之前的历史计算；
- 训练时 per-month 前向 OOF 自动使用当月之前的快照；
- 提交时 v15 引擎已固定截止到 2010-10-31，与历史数据自然衔接。
- 描述关键词从 train.csv 提取，与 train 共用同一窗口，无未来信息。
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# 简单的描述关键词提取（避免再发明 NLP）。分词规则：>=3 字母的英文词。
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def tokenize_description(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {w.lower() for w in _WORD_RE.findall(text)}


def _user_recent_invoices(history: pd.DataFrame, last_n: int = 3) -> dict:
    """返回 {user_id: list of (invoice_time, list(item_ids), mean_price, quantity_sum)}，
    每用户按时间倒序取最近 last_n 个 invoice。
    """
    history = history.loc[
        (history.Quantity > 0) & (history.UnitPrice > 0),
        ["user_id", "invoice_no", "item_id", "Quantity", "UnitPrice", "transaction_time"],
    ]
    price_amount = (history.UnitPrice * history.Quantity).astype(float)
    # 每 invoice 的时间与基础统计
    invoice_stats = history.groupby(["user_id", "invoice_no"], sort=False).agg(
        time=("transaction_time", "max"),
        items=("item_id", lambda s: list(s.astype(str).unique())),
        price_sum=("UnitPrice", lambda s: float(price_amount.loc[s.index].sum())),
        qty_sum=("Quantity", "sum"),
    )
    invoice_stats["mean_price"] = invoice_stats.price_sum / invoice_stats.qty_sum.clip(lower=1e-9)
    # 按用户按时间倒序
    invoice_stats = invoice_stats.sort_values(["user_id", "time"], ascending=[True, False])
    out: dict[str, list] = {}
    for uid, group in invoice_stats.groupby("user_id", sort=False):
        out[str(uid)] = [
            (
                row.time,
                list(row["items"]),
                float(row.mean_price),
                float(row.qty_sum),
            )
            for _, row in group.head(last_n).iterrows()
        ]
    return out


def _item_cooccurrence(history: pd.DataFrame) -> dict:
    """用 sparse 矩阵算 item-item 共现：
       - `mat` 是 (N_items, N_items) 共现矩阵，mat[a, b] = 同时买 a, b 的用户订单数
       - `row` 是 item_id → row 索引，`col` 同理
    调用时 P(b|a) = mat[a, b] / item_count[a]，item_count = mat.sum(axis=1)
    """
    from scipy.sparse import csr_matrix

    items_in_history = history.loc[
        (history.Quantity > 0) & (history.UnitPrice > 0),
        ["user_id", "invoice_no", "item_id"],
    ].drop_duplicates(["user_id", "invoice_no", "item_id"])

    # 编码
    unique_items = items_in_history.item_id.astype(str).unique()
    item_to_idx = {it: i for i, it in enumerate(unique_items)}
    items_in_history = items_in_history.assign(
        item_idx=items_in_history.item_id.astype(str).map(item_to_idx)
    )

    # 每个 (user, invoice) 一组 item_idx，求每组的两两组合
    inv_to_items = items_in_history.groupby(["user_id", "invoice_no"], sort=False).item_idx.apply(list)
    rows, cols = [], []
    for idx_list in inv_to_items:
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a, b = idx_list[i], idx_list[j]
                rows.append(a); cols.append(b)
                rows.append(b); cols.append(a)
    rows_arr = np.asarray(rows, dtype=np.int32)
    cols_arr = np.asarray(cols, dtype=np.int32)
    data = np.ones(len(rows_arr), dtype=np.float32)
    n_items = len(unique_items)
    mat = csr_matrix((data, (rows_arr, cols_arr)), shape=(n_items, n_items))
    # 合并重复并归一化：P(b|a) = coocc[a, b] / coocc[a, a] (coocc[a, a] = 有 a 的订单数)
    item_count = np.asarray(mat.sum(axis=1)).reshape(-1)
    # 避免除以 0
    safe_count = np.maximum(item_count, 1.0)
    inv_count = 1.0 / safe_count
    # 列归一化：把每一列 (b) 的值除以 count[a]，得到 P(b|a) 沿 a 索引
    diag = csr_matrix((inv_count, (np.arange(n_items), np.arange(n_items))),
                      shape=(n_items, n_items))
    prob_mat = diag @ mat
    prob_mat = prob_mat.tocsr()
    return {
        "item_to_idx": item_to_idx,
        "idx_to_item": {i: it for it, i in item_to_idx.items()},
        "prob_mat": prob_mat,
        "item_count": {item_to_idx[it]: float(item_count[item_to_idx[it]]) for it in item_to_idx},
    }


def _item_categories(history: pd.DataFrame) -> dict[str, set[str]]:
    """返回 {item_id: set(token)}，用 description 简单分词。"""
    desc = history[["item_id", "description"]].dropna(subset=["description"])
    desc["item_id"] = desc.item_id.astype(str)
    out: dict[str, set[str]] = {}
    for item, group in desc.groupby("item_id", sort=False):
        tokens: set[str] = set()
        for d in group.description:
            tokens.update(tokenize_description(d))
        out[item] = tokens
    return out


def build_v20_state(history: pd.DataFrame, last_n_invoices: int = 3) -> dict:
    """一次性计算 v20 全部静态状态。"""
    required = {"user_id", "item_id", "invoice_no", "Quantity", "UnitPrice",
                "transaction_time", "description"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history 缺列 {sorted(missing)}")
    user_recent = _user_recent_invoices(history, last_n=last_n_invoices)
    coocc = _item_cooccurrence(history)
    cats = _item_categories(history)
    return {
        "user_recent": user_recent,
        "coocc": coocc,
        "item_categories": cats,
    }


# ---- 候选打分：每对 (user, item) 算 6 列特征 ----

def _cooc_score(candidate: str, recent_items,
                coocc: dict) -> tuple[float, float]:
    if not recent_items:
        return 0.0, 0.0
    item_to_idx = coocc["item_to_idx"]
    cand_idx = item_to_idx.get(candidate)
    if cand_idx is None:
        return 0.0, 0.0
    recent_idx = [item_to_idx[a] for a in recent_items if a in item_to_idx]
    if not recent_idx:
        return 0.0, 0.0
    # prob_mat[cand_idx, recent_idx] = P(candidate | recent_i)
    row = coocc["prob_mat"].getrow(cand_idx)
    vals = np.asarray(row[:, recent_idx].todense()).reshape(-1)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0, 0.0
    return float(vals.max()), float(vals.mean())


def _price_gap(candidate_price: float, recent_price: float) -> float:
    if recent_price <= 0:
        return 0.0
    return abs(candidate_price - recent_price) / recent_price


def _category_overlap(candidate: str, recent_items,
                      item_cats: dict) -> tuple[float, float]:
    cand_set = item_cats.get(candidate, set())
    if not cand_set:
        return 0.0, 0.0
    overlaps = []
    for a in recent_items:
        a_set = item_cats.get(a, set())
        if not a_set:
            continue
        inter = len(cand_set & a_set)
        union = len(cand_set | a_set)
        if union > 0:
            overlaps.append(inter / union)
    if not overlaps:
        return 0.0, 0.0
    return float(max(overlaps)), float(sum(overlaps) / len(overlaps))


def _v20_columns_for_pair(uid: str, item: str, user_recent: dict,
                          coocc: dict,
                          item_cats: dict, item_price: dict) -> list[float]:
    invoices = user_recent.get(uid, [])
    if not invoices:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cand_price = item_price.get(item, 0.0)
    cooc_max, cooc_mean, price_max, price_mean, cat_max, cat_mean = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    for _, items, mean_price, _qty in invoices:
        item_set = set(items)
        if item in item_set:
            # 候选就是该 invoice 的实际商品
            cm, _ = _cooc_score(item, item_set - {item}, coocc)
            cooc_max = max(cooc_max, cm)
            cooc_mean = max(cooc_mean, cm)
            cat_m, _ = _category_overlap(item, item_set - {item}, item_cats)
            cat_max = max(cat_max, cat_m)
            cat_mean = max(cat_mean, cat_m)
            continue
        cm, cn = _cooc_score(item, item_set, coocc)
        cooc_max = max(cooc_max, cm)
        cooc_mean = max(cooc_mean, cn)
        price_gap = _price_gap(cand_price, mean_price)
        price_max = max(price_max, price_gap)
        price_mean = price_mean + price_gap
        cat_m, cat_n = _category_overlap(item, item_set, item_cats)
        cat_max = max(cat_max, cat_m)
        cat_mean = max(cat_mean, cat_n)
    n = len(invoices)
    price_mean = price_mean / max(1, n)
    cat_mean = cat_mean / max(1, n)
    return [cooc_max, cooc_mean, price_max, price_mean, cat_max, cat_mean]


def append_v20_frame(frame: pd.DataFrame, state: dict, item_price: dict) -> pd.DataFrame:
    """给整表追加 v20 6 列。frame 需含 user_id, item_id。item_price 来自 v9 引擎。"""
    if "user_id" not in frame.columns or "item_id" not in frame.columns:
        raise ValueError("frame 必须含 user_id 与 item_id")
    user_recent = state["user_recent"]
    coocc = state["coocc"]
    item_cats = state["item_categories"]
    rows = [
        _v20_columns_for_pair(
            str(u), str(i), user_recent, coocc, item_cats, item_price
        )
        for u, i in zip(frame["user_id"].astype(str), frame["item_id"].astype(str))
    ]
    cols = [
        "pair_basket_cooccurrence_max_3",
        "pair_basket_cooccurrence_mean_3",
        "pair_basket_price_gap_max_3",
        "pair_basket_price_gap_mean_3",
        "pair_basket_category_overlap_max_3",
        "pair_basket_category_overlap_mean_3",
    ]
    out = frame.copy()
    for name, idx in zip(cols, range(6)):
        out[name] = [r[idx] for r in rows]
    return out


def add_v20_columns(base: pd.DataFrame, state: dict, uid, item_ids, item_price: dict) -> pd.DataFrame:
    """提交期逐用户追加 v20 特征：base 是空表（index 与 item_ids 等长），返回 6 列 DataFrame。"""
    length = len(base)
    if length != len(item_ids):
        raise ValueError("base 与 item_ids 长度不一致")
    user_recent = state["user_recent"]
    coocc = state["coocc"]
    item_cats = state["item_categories"]
    rows = [
        _v20_columns_for_pair(
            str(uid), str(i), user_recent, coocc, item_cats, item_price
        )
        for i in item_ids
    ]
    return pd.DataFrame(rows, index=base.index, columns=[
        "pair_basket_cooccurrence_max_3",
        "pair_basket_cooccurrence_mean_3",
        "pair_basket_price_gap_max_3",
        "pair_basket_price_gap_mean_3",
        "pair_basket_category_overlap_max_3",
        "pair_basket_category_overlap_mean_3",
    ])
