"""v15 的时间安全交互特征与两阶段模型公共约定。

本模块只对已经由历史快照生成的列做确定性变换，不读取标签、候选文件或
未来交易。训练与提交共同调用 :func:`append_v15_interactions`，避免两条路径
出现同名异义或列顺序漂移。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


INTERACTION_FEATURES_V15 = [
    "user_recent_momentum",
    "item_recent_momentum",
    "season_user_recent_interaction",
    "season_item_growth_interaction",
    "season_pair_recent_interaction",
    "repurchase_due_recent_interaction",
]

USER_ACTIVITY_FEATURES = [
    "user_log_orders",
    "user_log_items",
    "user_log_recency_days",
    "user_log_recent30_orders",
    "user_log_recent90_orders",
    "user_recent30_order_share",
    "user_log_active30_days",
    "user_log_mean_price",
    "user_log_mean_order_amount",
    "user_log_typical_order_gap",
    "candidate_user_historical_positive_rate",
    "candidate_user_log_group_size",
    "candidate_user_prior_bought_fraction",
    "user_dec09_log_orders",
    "user_dec09_intensity",
    "user_dec09_active",
    "user_recent_momentum",
    "season_user_recent_interaction",
]


def append_v15_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """追加近期变化与季节性的交互项，保持输入索引和行顺序。"""
    required = {
        "user_log_recent30_orders", "user_log_recent90_orders",
        "item_log_recent30_orders", "item_log_recent90_orders",
        "item_recent_growth", "user_dec09_active", "item_dec09_intensity",
        "pair_dec09_bought", "user_item_log_recent90_orders",
        "user_item_repurchase_due_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"v15 交互特征缺少基础列：{sorted(missing)}")

    result = frame.copy()
    user30 = np.expm1(result["user_log_recent30_orders"].to_numpy(float))
    user90 = np.expm1(result["user_log_recent90_orders"].to_numpy(float))
    item30 = np.expm1(result["item_log_recent30_orders"].to_numpy(float))
    item90 = np.expm1(result["item_log_recent90_orders"].to_numpy(float))
    pair90 = np.expm1(result["user_item_log_recent90_orders"].to_numpy(float))

    result["user_recent_momentum"] = np.log1p(user30) - np.log1p(user90 / 3.0)
    result["item_recent_momentum"] = np.log1p(item30) - np.log1p(item90 / 3.0)
    result["season_user_recent_interaction"] = (
        result["user_dec09_active"].to_numpy(float) * np.log1p(user30)
    )
    result["season_item_growth_interaction"] = (
        result["item_dec09_intensity"].to_numpy(float)
        * result["item_recent_growth"].to_numpy(float)
    )
    result["season_pair_recent_interaction"] = (
        result["pair_dec09_bought"].to_numpy(float) * np.log1p(pair90)
    )
    result["repurchase_due_recent_interaction"] = (
        result["user_item_repurchase_due_ratio"].to_numpy(float)
        * np.log1p(pair90)
    )
    return result


def user_activity_frame(
    frame: pd.DataFrame, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """将候选行压成每个 ``snapshot_month,user_id`` 一行的用户表。"""
    selected = list(feature_names or USER_ACTIVITY_FEATURES)
    required = {"snapshot_month", "user_id", *selected}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"用户活跃度表缺少列：{sorted(missing)}")
    # 这些列按设计在同一用户快照内恒定；同时校验，避免静默取错行。
    grouped = frame.groupby(["snapshot_month", "user_id"], sort=False)
    nunique = grouped[selected].nunique(dropna=False)
    bad = nunique.gt(1).any(axis=1)
    if bad.any():
        examples = [tuple(value) for value in bad[bad].index[:5]]
        raise ValueError(f"用户活跃特征在组内不恒定，示例：{examples}")
    return grouped[selected].first().reset_index()
