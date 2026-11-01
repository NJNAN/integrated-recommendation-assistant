"""v16 用户月度活跃时序特征。

所有统计都按 ``snapshot_month`` 的月初截断，标签月及之后的交易不会参与。
训练与最终预测调用同一个函数，最终预测月份固定为 2010-11。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


USER_TEMPORAL_FEATURES_V16 = [
    "user_active_m1",
    "user_active_m2",
    "user_active_m3",
    "user_log_orders_m1",
    "user_log_orders_m2",
    "user_log_orders_m3",
    "user_active_months_6m",
    "user_recent3m_order_trend",
    "user_inactive_month_streak",
    "user_order_gap_cv",
    "user_due_ratio",
    "user_due_score",
]


def _state_for_month(clean: pd.DataFrame, month: str) -> pd.DataFrame:
    start = pd.Period(month, freq="M").start_time
    history = clean.loc[clean.transaction_time < start]
    orders = (history.groupby(["user_id", "invoice_no"], as_index=False)
              .transaction_time.max().sort_values(["user_id", "transaction_time"]))
    orders["month"] = orders.transaction_time.dt.to_period("M")
    monthly = orders.groupby(["user_id", "month"]).size().rename("orders").reset_index()
    users = pd.Index(history.user_id.astype(str).unique(), name="user_id")
    state = pd.DataFrame(index=users)
    target = pd.Period(month, freq="M")

    month_maps = {}
    for lag in range(1, 7):
        period = target - lag
        values = monthly.loc[monthly.month == period].set_index("user_id").orders.astype(float)
        month_maps[lag] = values
        if lag <= 3:
            count = state.index.to_series().map(values).fillna(0.0).to_numpy(float)
            state[f"user_active_m{lag}"] = (count > 0).astype(float)
            state[f"user_log_orders_m{lag}"] = np.log1p(count)
    active6 = np.zeros(len(state), dtype=float)
    for lag in range(1, 7):
        active6 += (state.index.to_series().map(month_maps[lag]).fillna(0).to_numpy(float) > 0)
    state["user_active_months_6m"] = active6

    m1 = np.expm1(state.user_log_orders_m1.to_numpy(float))
    m2 = np.expm1(state.user_log_orders_m2.to_numpy(float))
    m3 = np.expm1(state.user_log_orders_m3.to_numpy(float))
    state["user_recent3m_order_trend"] = np.log1p(m1) - np.log1p((m2 + m3) / 2.0)

    streak = np.zeros(len(state), dtype=float)
    still_inactive = np.ones(len(state), dtype=bool)
    for lag in range(1, 7):
        inactive = state.index.to_series().map(month_maps[lag]).fillna(0).to_numpy(float) == 0
        streak += (still_inactive & inactive).astype(float)
        still_inactive &= inactive
    state["user_inactive_month_streak"] = streak

    orders["gap_days"] = orders.groupby("user_id").transaction_time.diff().dt.total_seconds() / 86400.0
    gap = orders.dropna(subset=["gap_days"]).groupby("user_id").gap_days.agg(["median", "mean", "std"])
    gap_cv = (gap["std"].fillna(0.0) / gap["mean"].clip(lower=1.0)).clip(0, 5)
    last = orders.groupby("user_id").transaction_time.max()
    recency = (start - state.index.to_series().map(last)).dt.total_seconds().to_numpy(float) / 86400.0
    median_gap = state.index.to_series().map(gap["median"]).fillna(45.0).clip(lower=3.0).to_numpy(float)
    state["user_order_gap_cv"] = state.index.to_series().map(gap_cv).fillna(1.0).to_numpy(float)
    state["user_due_ratio"] = np.clip(recency / median_gap, 0, 12)
    width = 7.0 + 0.5 * median_gap
    state["user_due_score"] = np.exp(-np.abs(recency - median_gap) / width)
    return state.reset_index()


def append_user_temporal_features(frame: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """按月给候选行追加用户时序特征，并验证连接不增不减行。"""
    required = {"snapshot_month", "user_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"v16 时序特征缺少列：{sorted(missing)}")
    before = len(frame)
    pieces = []
    for month, group in frame.groupby("snapshot_month", sort=False):
        state = _state_for_month(clean, str(month))
        piece = group.merge(state, on="user_id", how="left", validate="many_to_one", sort=False)
        for name in USER_TEMPORAL_FEATURES_V16:
            piece[name] = piece[name].fillna(0.0)
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != before:
        raise ValueError("v16 用户时序特征连接改变了行数")
    return result
