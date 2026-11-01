"""比赛专用“季节/年度同比”特征（v11 新增模块，不改动任何现有代码）。

动机
----
v1-v9 迭代的结论是：训练-评分分布对齐带来的增益远大于换模型，且当前 57 维特征
全部来自“近期”（30/90 天窗口、recency、周期），没有任何一个特征能回答：

    这个 (user, item) 是不是“每年同一时期才买”的关系？

老师数据 train.csv 截止 2010-10-31，隐藏标签月极可能是 2010-11（圣诞采购季）。
v9 回放只覆盖 2010-06~10，模型从未见过一个圣诞月标签，因此对“去年圣诞买过、
夏天休眠、11 月会再买”的候选行会系统性压分——这不是噪声，是可被显式建模的
偏差。

本模块从 train.csv 的 2009-12（train 内唯一的圣诞月）与 2010-01/02 提取三个
粒度的年度信号：

* 商品侧：2009-12 订单数、买家数、12 月集中度、描述季节关键词；
* 用户侧：2009-12 订单数、12 月集中度、12 月是否活跃；
* 对（user, item）侧：2009-12 是否买过、订单数、数量。

防泄漏论证
----------
所有统计窗口严格截止于 2010-03-01，早于最早的候选回放月份（2010-06-01）。
因此这些特征值在全部回放折叠与最终打分时是**静态常量**，不会引入任何时间泄漏；
不依赖未来标签，也不依赖候选行本身（候选行只决定“给哪些行算特征”）。

用法
----
    from baseline.seasonal_features import build_seasonal_state, append_seasonal_frame

    state = build_seasonal_state(clean)          # clean 来自 load_training_data
    frame = append_seasonal_frame(frame, state)  # frame 需含 user_id / item_id 两列
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONAL_FEATURES = [
    "item_dec09_log_orders",
    "item_dec09_log_buyers",
    "item_dec09_intensity",
    "item_desc_seasonal",
    "user_dec09_log_orders",
    "user_dec09_intensity",
    "user_dec09_active",
    "pair_dec09_bought",
    "pair_dec09_log_orders",
    "pair_dec09_log_quantity",
]

# 圣诞/万圣/冬季相关关键词。'SEASON' 会命中 'SEASONAL'（如季节性装饰）。
SEASONAL_KEYWORDS = (
    "CHRISTMAS", "XMAS", "SANTA", "HALLOWEEN",
    "WINTER", "STOCKING", "SEASON",
)

DEC09_START = pd.Timestamp("2009-12-01")
DEC09_END = pd.Timestamp("2010-01-01")   # [2009-12-01, 2010-01-01) = 整个 2009-12
EARLY10_END = pd.Timestamp("2010-03-01")  # 对照窗 [2010-01-01, 2010-03-01)


def _keyword_flag(description) -> int:
    if not isinstance(description, str):
        return 0
    upper = description.upper()
    return int(any(keyword in upper for keyword in SEASONAL_KEYWORDS))


def build_seasonal_state(clean: pd.DataFrame) -> dict:
    """从清洗交易表一次性计算全部年度静态统计。

    clean 必须包含 user_id / item_id / invoice_no / transaction_time / Quantity
    列（load_training_data 的输出即满足），transaction_time 为 datetime64。
    返回的 state 可 pickle 保存，供训练与提交复用，保证两侧特征完全一致。
    """
    required = {"user_id", "item_id", "invoice_no", "transaction_time", "Quantity"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"clean 缺少列：{sorted(missing)}")

    time = clean.transaction_time
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("transaction_time 必须为 datetime64")

    dec09 = clean.loc[(time >= DEC09_START) & (time < DEC09_END)].copy()
    early10 = clean.loc[(time >= DEC09_END) & (time < EARLY10_END)].copy()
    del time

    # —— 商品侧：2009-12 订单数 / 买家数，2010-01/02 订单数（作集中度分母）——
    item_dec = dec09.groupby("item_id").agg(
        orders=("invoice_no", "nunique"), buyers=("user_id", "nunique"),
    )
    item_early_orders = early10.groupby("item_id").invoice_no.nunique().rename("orders")

    # —— 用户侧：2009-12 订单数，2010-01/02 订单数 ——
    user_dec = dec09.groupby("user_id").invoice_no.nunique().rename("orders")
    user_early = early10.groupby("user_id").invoice_no.nunique().rename("orders")

    # —— 对（user, item）侧：2009-12 订单数 / 数量 ——
    pair_dec = dec09.groupby(["user_id", "item_id"]).agg(
        orders=("invoice_no", "nunique"), quantity=("Quantity", "sum"),
    )
    pair_dec["quantity"] = pair_dec.quantity.clip(lower=0.0)
    # 键统一为 "user\x1fitem" 拼接串，与 _feature_values 的查找键完全一致。
    pair_key = (
        pair_dec.index.get_level_values(0).astype(str).str.cat(
            pair_dec.index.get_level_values(1).astype(str), sep="\x1f"
        )
    )

    # —— 商品描述关键词（来自清洗表自带 description 列，缺失视为无关键词）——
    descriptions = (
        clean[["item_id", "description"]]
        .dropna(subset=["description"])
        .groupby("item_id").description.first()
    )

    def intensity(dec_series: pd.Series, early_series: pd.Series) -> pd.Series:
        dec = dec_series.reindex(dec_series.index).fillna(0.0)
        denominator = dec.add(early_series.reindex(dec_series.index).fillna(0.0))
        return np.where(denominator > 0, dec / np.maximum(denominator, 1e-9), 0.0)

    state = {
        "item_dec09_orders": item_dec.orders.astype(float).to_dict(),
        "item_dec09_buyers": item_dec.buyers.astype(float).to_dict(),
        "item_dec09_intensity": dict(
            zip(item_dec.index, intensity(item_dec.orders, item_early_orders))
        ),
        "item_desc_seasonal": {
            str(item): _keyword_flag(description)
            for item, description in descriptions.items()
        },
        "user_dec09_orders": user_dec.astype(float).to_dict(),
        "user_dec09_intensity": dict(
            zip(user_dec.index, intensity(user_dec, user_early))
        ),
        "pair_dec09_orders": {
            key: float(value)
            for key, value in zip(pair_key, pair_dec.orders.astype(float))
        },
        "pair_dec09_quantity": {
            key: float(value)
            for key, value in zip(pair_key, pair_dec.quantity.astype(float))
        },
    }
    return state


def _feature_values(users: pd.Series, items: pd.Series, state: dict) -> dict:
    """对任意等长 user/item 序列计算全部季节特征（向量化，共享给两条入口）。"""
    users = users.astype(str).reset_index(drop=True)
    items = items.astype(str).reset_index(drop=True)
    keys = users.str.cat(items, sep="\x1f")

    pair_orders = keys.map(state["pair_dec09_orders"]).fillna(0.0).to_numpy(float)
    pair_quantity = keys.map(state["pair_dec09_quantity"]).fillna(0.0).to_numpy(float)

    item_orders = items.map(state["item_dec09_orders"]).fillna(0.0).to_numpy(float)
    item_buyers = items.map(state["item_dec09_buyers"]).fillna(0.0).to_numpy(float)
    item_intensity = items.map(state["item_dec09_intensity"]).fillna(0.0).to_numpy(float)
    item_seasonal = items.map(state["item_desc_seasonal"]).fillna(0).to_numpy(float)

    user_orders = users.map(state["user_dec09_orders"]).fillna(0.0).to_numpy(float)
    user_intensity = users.map(state["user_dec09_intensity"]).fillna(0.0).to_numpy(float)

    return {
        "item_dec09_log_orders": np.log1p(item_orders),
        "item_dec09_log_buyers": np.log1p(item_buyers),
        "item_dec09_intensity": item_intensity,
        "item_desc_seasonal": item_seasonal,
        "user_dec09_log_orders": np.log1p(user_orders),
        "user_dec09_intensity": user_intensity,
        "user_dec09_active": (user_orders > 0).astype(float),
        "pair_dec09_bought": (pair_orders > 0).astype(float),
        "pair_dec09_log_orders": np.log1p(pair_orders),
        "pair_dec09_log_quantity": np.log1p(pair_quantity),
    }


def append_seasonal_frame(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    """给整表追加季节特征列（frame 需含 user_id / item_id），返回副本。"""
    if "user_id" not in frame.columns or "item_id" not in frame.columns:
        raise ValueError("frame 必须包含 user_id 与 item_id 两列")
    values = _feature_values(frame["user_id"], frame["item_id"], state)
    result = frame.copy()
    for name in SEASONAL_FEATURES:
        result[name] = values[name]
    return result


def add_seasonal_columns(base: pd.DataFrame, state: dict, uid, item_ids) -> pd.DataFrame:
    """提交期逐用户追加季节特征：base 是单用户特征表，item_ids 与之等长。

    返回与 base 索引对齐、仅含 SEASONAL_FEATURES 的 DataFrame，供拼接到
    competition_features 输出之后，保证训练/提交两条路径列名与含义完全一致。
    """
    length = len(base)
    if length != len(item_ids):
        raise ValueError("base 与 item_ids 长度不一致")
    values = _feature_values(
        pd.Series([str(uid)] * length), pd.Series(list(item_ids)), state
    )
    return pd.DataFrame(
        {name: values[name] for name in SEASONAL_FEATURES}, index=base.index
    )
