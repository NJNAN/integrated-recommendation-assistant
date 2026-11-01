"""v13 月度动态附加特征（在每月快照引擎上计算，与 v9/v11 的静态季节特征互补）。

动机：v9/v11 的 57+10 个特征全部来自用户/商品/交互的计数与时效统计，缺少
三个“价格与篮子”维度的信息：

1. pair_price_gap_ratio ：(user,item) 最近一次成交单价 / 该商品历史中位价。
   批发场景里价格异动（折扣、涨价）会直接改变补货概率；
2. item_price_trend      ：商品近 90 天中位价 / 全历史中位价的对数比。
   反映该商品近期价格环境变化（季节清仓/调价）；
3. co_purchase_recent    ：用户近期购买过的商品中，有多少与候选商品在历史上
   同单出现（log1p 求和）。补货通常是整篮行为，不只复购单品。

防泄漏：全部状态在“打分时点之前”的交易上计算（attach 时 history 已经按
月份截止），近 90 天窗口随快照月份自然前移；提交期使用截止 2010-10-31 的
全历史引擎，语义与训练一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRA_FEATURES_V13 = [
    "pair_price_gap_ratio",
    "item_price_trend",
    "co_purchase_recent_hits",
]

# 共购近邻保留数量：越大越精确，但 attach 状态与内存线性增长
CO_BUY_TOP_N = 40


def attach_v13_state(engine, history: pd.DataFrame, recent_items: int = 20) -> None:
    """把价格与共购状态挂到引擎上，pickle 保存时一并写入。"""
    reference = pd.Timestamp(history.transaction_time.max()) + pd.Timedelta(seconds=1)

    # —— 1. pair 最近成交单价 vs 商品全历史中位价 ——
    pair_price = (
        history.sort_values("transaction_time")
        .groupby(["user_id", "item_id"])["UnitPrice"]
        .last()
    )
    item_price_median = history.groupby("item_id")["UnitPrice"].median()
    pair_key = pair_price.index.get_level_values(0).astype(str).str.cat(
        pair_price.index.get_level_values(1).astype(str), sep="\x1f"
    )
    engine.v13_pair_last_price = {
        key: float(value)
        for key, value in zip(pair_key, pair_price.astype(float))
    }
    engine.v13_item_price_median = item_price_median.astype(float).to_dict()

    # —— 2. item 近 90 天中位价 vs 全历史中位价（对数比）——
    recent90 = history.loc[
        history.transaction_time >= reference - pd.Timedelta(days=90)
    ]
    item_price_recent = recent90.groupby("item_id")["UnitPrice"].median()
    engine.v13_item_price_trend = {
        str(item): float(value)
        for item, value in item_price_recent.items()
        if item in item_price_median.index and item_price_median[item] > 0
    }

    # —— 3. 共购近邻：同单出现次数 top-N（用户近期商品 ↔ 候选商品）——
    order_items = history[["invoice_no", "item_id"]].drop_duplicates()
    merged = order_items.merge(order_items, on="invoice_no", suffixes=("_a", "_b"))
    merged = merged.loc[merged.item_id_a != merged.item_id_b]
    counts = merged.groupby(["item_id_a", "item_id_b"]).size()
    engine.v13_cobuy_top = {}
    for item, group in counts.groupby(level=0):
        top = group.sort_values(ascending=False).head(CO_BUY_TOP_N)
        engine.v13_cobuy_top[str(item)] = {
            str(nb): float(value) for nb, value in top.items()
        }
    engine.v13_recent_items = recent_items


def v13_feature_values(engine, uid: str, item_ids) -> dict[str, np.ndarray]:
    """对单用户候选列表计算 v13 特征（与 competition_features 同模式）。"""
    if not hasattr(engine, "v13_pair_last_price"):
        raise ValueError("引擎缺少 v13 状态，请先运行 attach_v13_state")

    uid = str(uid)
    item_ids = [str(item) for item in item_ids]
    pair_keys = [f"{uid}\x1f{item}" for item in item_ids]

    pair_price = np.asarray(
        [engine.v13_pair_last_price.get(key, 0.0) for key in pair_keys], dtype=float
    )
    item_median = np.asarray(
        [engine.v13_item_price_median.get(item, 0.0) for item in item_ids], dtype=float
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(
            (pair_price > 0) & (item_median > 0),
            pair_price / np.maximum(item_median, 1e-9),
            0.0,
        )
    gap_log = np.where(gap > 0, np.log(np.clip(gap, 0.25, 4.0)), 0.0)

    trend = np.asarray(
        [engine.v13_item_price_trend.get(item, 0.0) for item in item_ids], dtype=float
    )
    median = np.asarray(
        [engine.v13_item_price_median.get(item, 0.0) for item in item_ids], dtype=float
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_value = np.where(
            (trend > 0) & (median > 0),
            np.log(np.clip(trend / np.maximum(median, 1e-9), 0.25, 4.0)),
            0.0,
        )

    recent = [
        str(item) for item in engine.histories.get(uid, [])[-engine.v13_recent_items:]
    ]
    cobuy = np.zeros(len(item_ids), dtype=float)
    if recent:
        for index, item in enumerate(item_ids):
            top = engine.v13_cobuy_top.get(item, {})
            hits = sum(top.get(nb, 0.0) for nb in recent)
            cobuy[index] = hits

    return {
        "pair_price_gap_ratio": gap_log,
        "item_price_trend": trend_value,
        "co_purchase_recent_hits": np.log1p(cobuy),
    }


def append_v13_columns(base: pd.DataFrame, engine, uid, item_ids) -> pd.DataFrame:
    """返回与 base 索引对齐、仅含 EXTRA_FEATURES_V13 的 DataFrame。"""
    if len(base) != len(item_ids):
        raise ValueError("base 与 item_ids 长度不一致")
    values = v13_feature_values(engine, uid, item_ids)
    return pd.DataFrame(
        {name: values[name] for name in EXTRA_FEATURES_V13}, index=base.index
    )
