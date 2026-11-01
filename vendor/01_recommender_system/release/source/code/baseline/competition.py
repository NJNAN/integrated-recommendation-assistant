"""比赛专用的用户—商品交叉特征。

这些特征只从打分时点之前的交易计算，并随比赛引擎一起持久化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .engine import FEATURES


EXTRA_FEATURES = [
    "prior_bought",
    "user_item_log_orders",
    "user_item_log_quantity",
    "user_item_log_recency_days",
    "user_item_order_share",
    "item_log_recent30_orders",
    "item_log_recent90_orders",
    "item_recent_growth",
    "item_log_recency_days",
    "user_log_recency_days",
    "user_item_log_recent30_orders",
    "user_item_log_recent90_orders",
    "user_item_repurchase_due_ratio",
    "user_item_log_active_months",
    "user_log_recent30_orders",
    "user_log_recent90_orders",
    "user_recent30_order_share",
    "user_log_active30_days",
    "user_log_mean_price",
    "user_log_mean_order_amount",
    "item_log_recent30_buyers",
    "item_log_recent30_quantity",
    "item_log_age_days",
    "price_gap_ratio",
    "candidate_item_log_frequency",
    "user_item_log_gap_count",
    "user_item_gap_cv",
    "user_item_cycle_distance",
    "user_item_days_past_due_ratio",
    "user_item_repurchase_hazard",
    "user_item_next30_cycle_score",
    "user_log_typical_order_gap",
    "item_log_typical_repurchase_gap",
    "time_decay_itemcf_score",
    "bpr_score",
    "bpr_cosine",
    "bpr_item_bias",
    "candidate_user_historical_positive_rate",
    "candidate_item_historical_positive_rate",
    "candidate_pair_active_month_rate",
    "candidate_pair_recent3m_positive_rate",
    "candidate_pair_frequency_lift",
    "candidate_user_log_group_size",
    "candidate_user_prior_bought_fraction",
]
COMPETITION_FEATURES = FEATURES + EXTRA_FEATURES


def attach_competition_state(
    engine,
    history: pd.DataFrame,
    candidate_item_weights=None,
    candidate_frame: pd.DataFrame | None = None,
) -> None:
    """把历史期 RFM 与趋势统计挂到引擎，pickle 保存时会一并写入。"""
    reference = pd.Timestamp(history.transaction_time.max()) + pd.Timedelta(seconds=1)
    engine.competition_reference_time = reference

    pairs = history.groupby(["user_id", "item_id"], sort=False).agg(
        orders=("invoice_no", "nunique"),
        quantity=("Quantity", "sum"),
        last_time=("transaction_time", "max"),
    )
    engine.competition_pair_orders = pairs.orders.astype(float).to_dict()
    engine.competition_pair_quantity = pairs.quantity.clip(lower=0).astype(float).to_dict()
    engine.competition_pair_last = pairs.last_time.to_dict()

    order_pairs = history.groupby(["user_id", "item_id", "invoice_no"], sort=False).agg(
        time=("transaction_time", "max"), quantity=("Quantity", "sum")
    ).reset_index().sort_values(["user_id", "item_id", "time"])
    order_pairs["gap_days"] = order_pairs.groupby(["user_id", "item_id"]).time.diff().dt.total_seconds() / 86400
    valid_pair_gaps = order_pairs.dropna(subset=["gap_days"])
    pair_gap = valid_pair_gaps.groupby(["user_id", "item_id"]).gap_days.agg(
        ["median", "mean", "std", "count"]
    )
    engine.competition_pair_median_gap = pair_gap["median"].astype(float).to_dict()
    engine.competition_pair_mean_gap = pair_gap["mean"].astype(float).to_dict()
    engine.competition_pair_std_gap = pair_gap["std"].fillna(0).astype(float).to_dict()
    engine.competition_pair_gap_count = pair_gap["count"].astype(float).to_dict()
    engine.competition_item_median_gap = valid_pair_gaps.groupby("item_id").gap_days.median().astype(float).to_dict()
    engine.competition_pair_active_months = order_pairs.assign(
        month=order_pairs.time.dt.to_period("M").astype(str)
    ).groupby(["user_id", "item_id"]).month.nunique().astype(float).to_dict()

    user_orders = history.groupby(["user_id", "invoice_no"], sort=False).transaction_time.max().reset_index()
    user_orders = user_orders.sort_values(["user_id", "transaction_time"])
    user_orders["gap_days"] = user_orders.groupby("user_id").transaction_time.diff().dt.total_seconds() / 86400
    engine.competition_user_median_gap = user_orders.dropna(subset=["gap_days"]).groupby(
        "user_id"
    ).gap_days.median().astype(float).to_dict()

    item_last = history.groupby("item_id").transaction_time.max()
    user_last = history.groupby("user_id").transaction_time.max()
    engine.competition_item_last = item_last.to_dict()
    engine.competition_user_last = user_last.to_dict()

    order_items = history[["user_id", "invoice_no", "item_id", "transaction_time"]].drop_duplicates(
        ["user_id", "invoice_no", "item_id"]
    )
    recent30 = order_items.loc[order_items.transaction_time >= reference - pd.Timedelta(days=30)]
    recent90 = order_items.loc[order_items.transaction_time >= reference - pd.Timedelta(days=90)]
    engine.competition_item_recent30 = recent30.groupby("item_id").invoice_no.nunique().astype(float).to_dict()
    engine.competition_item_recent90 = recent90.groupby("item_id").invoice_no.nunique().astype(float).to_dict()
    engine.competition_pair_recent30 = recent30.groupby(["user_id", "item_id"]).invoice_no.nunique().astype(float).to_dict()
    engine.competition_pair_recent90 = recent90.groupby(["user_id", "item_id"]).invoice_no.nunique().astype(float).to_dict()
    engine.competition_user_recent30 = recent30.groupby("user_id").invoice_no.nunique().astype(float).to_dict()
    engine.competition_user_recent90 = recent90.groupby("user_id").invoice_no.nunique().astype(float).to_dict()
    engine.competition_user_active30 = recent30.assign(
        day=recent30.transaction_time.dt.normalize()
    ).groupby("user_id").day.nunique().astype(float).to_dict()
    engine.competition_item_recent30_buyers = recent30.groupby("item_id").user_id.nunique().astype(float).to_dict()
    recent30_quantities = history.loc[
        history.transaction_time >= reference - pd.Timedelta(days=30)
    ].groupby("item_id").Quantity.sum()
    engine.competition_item_recent30_quantity = recent30_quantities.clip(lower=0).astype(float).to_dict()

    engine.competition_item_first = history.groupby("item_id").transaction_time.min().to_dict()
    engine.competition_item_median_price = history.groupby("item_id").UnitPrice.median().astype(float).to_dict()
    engine.competition_user_mean_price = history.groupby("user_id").UnitPrice.mean().astype(float).to_dict()
    amounts = history.assign(amount=history.Quantity * history.UnitPrice).groupby(
        ["user_id", "invoice_no"]
    ).amount.sum()
    engine.competition_user_mean_order_amount = amounts.groupby("user_id").mean().clip(lower=0).astype(float).to_dict()
    engine.competition_candidate_item_frequency = {
        str(item): float(value) for item, value in (candidate_item_weights or {}).items()
    }

    engine.competition_candidate_user_positive_rate = {}
    engine.competition_candidate_item_positive_rate = {}
    engine.competition_candidate_pair_month_rate = {}
    engine.competition_candidate_pair_recent3m_rate = {}
    engine.competition_candidate_pair_lift = {}
    engine.competition_candidate_user_group_size = {}
    engine.competition_candidate_user_prior_fraction = {}
    if candidate_frame is not None:
        fixed = candidate_frame[["user_id", "item_id"]].astype(str).drop_duplicates()
        first_month = pd.Period(history.transaction_time.min(), freq="M")
        last_month = pd.Period(history.transaction_time.max(), freq="M")
        observed_months = max(1, last_month.ordinal - first_month.ordinal + 1)
        monthly = history[["user_id", "item_id", "transaction_time"]].copy()
        monthly["user_id"] = monthly.user_id.astype(str)
        monthly["item_id"] = monthly.item_id.astype(str)
        monthly["month"] = monthly.transaction_time.dt.to_period("M").astype(str)
        monthly = monthly.merge(fixed, on=["user_id", "item_id"], how="inner")
        monthly = monthly.drop_duplicates(["month", "user_id", "item_id"])
        pair_months = monthly.groupby(["user_id", "item_id"]).month.nunique().astype(float)
        user_counts = fixed.groupby("user_id").item_id.nunique().astype(float)
        item_counts = fixed.groupby("item_id").user_id.nunique().astype(float)
        user_hits = monthly.groupby("user_id").size().astype(float)
        item_hits = monthly.groupby("item_id").size().astype(float)
        user_rates = (user_hits / (user_counts * observed_months)).fillna(0.0)
        item_rates = (item_hits / (item_counts * observed_months)).fillna(0.0)
        pair_rates = pair_months / observed_months
        recent_start = reference - pd.Timedelta(days=92)
        recent = monthly.loc[monthly.transaction_time >= recent_start]
        recent_pair = recent.groupby(["user_id", "item_id"]).month.nunique().astype(float) / 3.0
        engine.competition_candidate_user_positive_rate = user_rates.astype(float).to_dict()
        engine.competition_candidate_item_positive_rate = item_rates.astype(float).to_dict()
        engine.competition_candidate_pair_month_rate = pair_rates.astype(float).to_dict()
        engine.competition_candidate_pair_recent3m_rate = recent_pair.astype(float).to_dict()
        engine.competition_candidate_user_group_size = user_counts.astype(float).to_dict()
        bought_pairs = set(zip(
            history.user_id.astype(str), history.item_id.astype(str)
        ))
        prior = fixed.assign(
            prior=[(user, item) in bought_pairs for user, item in zip(fixed.user_id, fixed.item_id)]
        ).groupby("user_id").prior.mean()
        engine.competition_candidate_user_prior_fraction = prior.astype(float).to_dict()
        for key, pair_rate in pair_rates.items():
            user_rate = float(user_rates.get(key[0], 0.0))
            item_rate = float(item_rates.get(key[1], 0.0))
            denominator = max(1e-4, np.sqrt(max(user_rate, 1e-6) * max(item_rate, 1e-6)))
            engine.competition_candidate_pair_lift[key] = float(min(100.0, pair_rate / denominator))

    # Recency-weighted ItemCF.  It differs from the baseline binary ItemCF by
    # emphasizing recent and repeated user-item interactions.  Only top
    # neighbours are retained, so the final engine stays reasonably small.
    weighted_pairs = history.groupby(["user_id", "item_id"], sort=False).agg(
        last_time=("transaction_time", "max"), orders=("invoice_no", "nunique")
    ).reset_index()
    age_days = (reference - weighted_pairs.last_time).dt.total_seconds().to_numpy(float) / 86400.0
    values = np.exp(-np.maximum(age_days, 0.0) / 150.0) * (
        1.0 + 0.35 * np.log1p(weighted_pairs.orders.to_numpy(float))
    )
    matrix = csr_matrix(
        (values.astype(np.float32), (
            weighted_pairs.user_id.astype(str).map(engine.user_pos).to_numpy(),
            weighted_pairs.item_id.astype(str).map(engine.pos).to_numpy(),
        )),
        shape=(len(engine.users), len(engine.ids)),
        dtype=np.float32,
    )
    normalized = normalize(matrix, axis=0)
    similarity = (normalized.T @ normalized).toarray().astype(np.float32, copy=False)
    np.fill_diagonal(similarity, 0.0)
    engine.competition_time_decay_itemcf = engine._sparsify(
        similarity, min(90, max(30, engine.cfg.neighbors))
    )


def competition_features(engine, uid: str, candidates, scores) -> pd.DataFrame:
    """在原 13 维特征后追加复购、时效和趋势特征。"""
    if not hasattr(engine, "competition_pair_orders"):
        raise ValueError("当前引擎没有比赛交叉特征，请先运行 train_competition.py")
    uid = str(uid)
    base = engine.features(uid, candidates, scores)
    reference = engine.competition_reference_time
    unseen_recency = 3650.0
    user_orders = max(1.0, float(engine.orders.get(uid, 0)))

    candidate_items = [str(candidate["item_id"]) for candidate in candidates]
    candidate_positions = np.asarray([engine.pos[item] for item in candidate_items], dtype=int)
    recent_history = [
        engine.pos[item] for item in engine.histories.get(uid, [])[-engine.cfg.recent_items:]
        if item in engine.pos
    ]
    if recent_history and hasattr(engine, "competition_time_decay_itemcf"):
        time_cf = engine.competition_time_decay_itemcf[recent_history][:, candidate_positions]
        time_cf_scores = np.asarray(time_cf.sum(axis=0)).ravel() / np.sqrt(len(recent_history))
    else:
        time_cf_scores = np.zeros(len(candidate_items), dtype=float)

    bpr_user_index = getattr(engine, "competition_bpr_user_pos", {}).get(uid)
    bpr_item_map = getattr(engine, "competition_bpr_item_pos", {})
    if bpr_user_index is not None and hasattr(engine, "competition_bpr_user_factors"):
        bpr_user = engine.competition_bpr_user_factors[bpr_user_index]
        bpr_user_norm = max(float(np.linalg.norm(bpr_user)), 1e-8)
    else:
        bpr_user = None
        bpr_user_norm = 1.0

    columns = {name: [] for name in EXTRA_FEATURES}
    for candidate_index, candidate in enumerate(candidates):
        item = str(candidate["item_id"])
        key = (uid, item)
        pair_orders = float(engine.competition_pair_orders.get(key, 0.0))
        pair_quantity = float(engine.competition_pair_quantity.get(key, 0.0))
        pair_last = engine.competition_pair_last.get(key)
        item_last = engine.competition_item_last.get(item)
        user_last = engine.competition_user_last.get(uid)
        recent30 = float(engine.competition_item_recent30.get(item, 0.0))
        recent90 = float(engine.competition_item_recent90.get(item, 0.0))
        previous60_monthly = max(0.0, recent90 - recent30) / 2.0

        pair_recency = (reference - pair_last).total_seconds() / 86400 if pair_last is not None else unseen_recency
        item_recency = (reference - item_last).total_seconds() / 86400 if item_last is not None else unseen_recency
        user_recency = (reference - user_last).total_seconds() / 86400 if user_last is not None else unseen_recency
        pair_recent30 = float(engine.competition_pair_recent30.get(key, 0.0))
        pair_recent90 = float(engine.competition_pair_recent90.get(key, 0.0))
        median_gap = float(engine.competition_pair_median_gap.get(key, 0.0))
        mean_gap = float(getattr(engine, "competition_pair_mean_gap", {}).get(key, 0.0))
        std_gap = float(getattr(engine, "competition_pair_std_gap", {}).get(key, 0.0))
        gap_count = float(getattr(engine, "competition_pair_gap_count", {}).get(key, 0.0))
        user_typical_gap = float(getattr(engine, "competition_user_median_gap", {}).get(uid, 0.0))
        item_typical_gap = float(getattr(engine, "competition_item_median_gap", {}).get(item, 0.0))
        active_months = float(engine.competition_pair_active_months.get(key, 0.0))
        user_recent30 = float(engine.competition_user_recent30.get(uid, 0.0))
        user_recent90 = float(engine.competition_user_recent90.get(uid, 0.0))
        user_active30 = float(engine.competition_user_active30.get(uid, 0.0))
        user_mean_price = float(engine.competition_user_mean_price.get(uid, 0.0))
        user_mean_order_amount = float(engine.competition_user_mean_order_amount.get(uid, 0.0))
        item_recent30_buyers = float(engine.competition_item_recent30_buyers.get(item, 0.0))
        item_recent30_quantity = float(engine.competition_item_recent30_quantity.get(item, 0.0))
        item_first = engine.competition_item_first.get(item)
        item_age = (reference - item_first).total_seconds() / 86400 if item_first is not None else 0.0
        item_price = float(engine.competition_item_median_price.get(item, 0.0))
        candidate_frequency = float(engine.competition_candidate_item_frequency.get(item, 0.0))
        candidate_user_rate = float(getattr(
            engine, "competition_candidate_user_positive_rate", {}
        ).get(uid, 0.0))
        candidate_item_rate = float(getattr(
            engine, "competition_candidate_item_positive_rate", {}
        ).get(item, 0.0))
        candidate_pair_rate = float(getattr(
            engine, "competition_candidate_pair_month_rate", {}
        ).get(key, 0.0))
        candidate_pair_recent3m_rate = float(getattr(
            engine, "competition_candidate_pair_recent3m_rate", {}
        ).get(key, 0.0))
        candidate_pair_lift = float(getattr(
            engine, "competition_candidate_pair_lift", {}
        ).get(key, 0.0))
        candidate_group_size = float(getattr(
            engine, "competition_candidate_user_group_size", {}
        ).get(uid, 0.0))
        candidate_user_prior_fraction = float(getattr(
            engine, "competition_candidate_user_prior_fraction", {}
        ).get(uid, 0.0))

        # Shrink sparse pair cadence towards item and user cadence.  A pair with
        # multiple observed gaps is trusted; a once-bought pair falls back to
        # broad user/item purchasing rhythms rather than receiving all zeros.
        fallback_gaps = [value for value in (item_typical_gap, user_typical_gap) if value > 0]
        fallback_gap = float(np.median(fallback_gaps)) if fallback_gaps else 45.0
        pair_typical = median_gap if median_gap > 0 else (mean_gap if mean_gap > 0 else fallback_gap)
        reliability = min(1.0, gap_count / 3.0)
        typical_gap = reliability * pair_typical + (1.0 - reliability) * fallback_gap
        spread = max(7.0, std_gap if std_gap > 0 else typical_gap * 0.45)
        cycle_distance = abs(pair_recency - typical_gap) / max(7.0, typical_gap)
        days_past_due = (pair_recency - typical_gap) / max(7.0, typical_gap)
        repurchase_hazard = np.exp(-0.5 * ((pair_recency - typical_gap) / spread) ** 2) if pair_orders else 0.0
        # Approximate whether the next expected replenishment falls within the
        # coming 30 days.  This is intentionally a smooth feature, not a label.
        next30_distance = max(0.0, typical_gap - pair_recency)
        next30_cycle_score = (
            np.exp(-next30_distance / max(7.0, spread)) * np.exp(-max(0.0, pair_recency - typical_gap) / max(14.0, 2 * spread))
            if pair_orders else 0.0
        )

        bpr_item_index = bpr_item_map.get(item)
        if bpr_user is not None and bpr_item_index is not None:
            bpr_item = engine.competition_bpr_item_factors[bpr_item_index]
            bpr_bias = float(engine.competition_bpr_item_bias[bpr_item_index])
            bpr_dot = float(np.dot(bpr_user, bpr_item))
            bpr_cosine = bpr_dot / (bpr_user_norm * max(float(np.linalg.norm(bpr_item)), 1e-8))
            bpr_value = bpr_dot + bpr_bias
        else:
            bpr_value = bpr_cosine = bpr_bias = 0.0

        columns["prior_bought"].append(float(pair_orders > 0))
        columns["user_item_log_orders"].append(np.log1p(pair_orders))
        columns["user_item_log_quantity"].append(np.log1p(pair_quantity))
        columns["user_item_log_recency_days"].append(np.log1p(max(0.0, pair_recency)))
        columns["user_item_order_share"].append(pair_orders / user_orders)
        columns["item_log_recent30_orders"].append(np.log1p(recent30))
        columns["item_log_recent90_orders"].append(np.log1p(recent90))
        columns["item_recent_growth"].append(np.log1p(recent30) - np.log1p(previous60_monthly))
        columns["item_log_recency_days"].append(np.log1p(max(0.0, item_recency)))
        columns["user_log_recency_days"].append(np.log1p(max(0.0, user_recency)))
        columns["user_item_log_recent30_orders"].append(np.log1p(pair_recent30))
        columns["user_item_log_recent90_orders"].append(np.log1p(pair_recent90))
        columns["user_item_repurchase_due_ratio"].append(
            pair_recency / median_gap if median_gap > 0 else 0.0
        )
        columns["user_item_log_active_months"].append(np.log1p(active_months))
        columns["user_log_recent30_orders"].append(np.log1p(user_recent30))
        columns["user_log_recent90_orders"].append(np.log1p(user_recent90))
        columns["user_recent30_order_share"].append(user_recent30 / user_orders)
        columns["user_log_active30_days"].append(np.log1p(user_active30))
        columns["user_log_mean_price"].append(np.log1p(max(0.0, user_mean_price)))
        columns["user_log_mean_order_amount"].append(np.log1p(max(0.0, user_mean_order_amount)))
        columns["item_log_recent30_buyers"].append(np.log1p(item_recent30_buyers))
        columns["item_log_recent30_quantity"].append(np.log1p(item_recent30_quantity))
        columns["item_log_age_days"].append(np.log1p(max(0.0, item_age)))
        columns["price_gap_ratio"].append(
            abs(item_price - user_mean_price) / max(0.01, user_mean_price)
        )
        columns["candidate_item_log_frequency"].append(np.log1p(candidate_frequency))
        columns["user_item_log_gap_count"].append(np.log1p(gap_count))
        columns["user_item_gap_cv"].append(std_gap / max(1.0, mean_gap) if mean_gap > 0 else 0.0)
        columns["user_item_cycle_distance"].append(cycle_distance if pair_orders else 0.0)
        columns["user_item_days_past_due_ratio"].append(days_past_due if pair_orders else 0.0)
        columns["user_item_repurchase_hazard"].append(repurchase_hazard)
        columns["user_item_next30_cycle_score"].append(next30_cycle_score)
        columns["user_log_typical_order_gap"].append(np.log1p(max(0.0, user_typical_gap)))
        columns["item_log_typical_repurchase_gap"].append(np.log1p(max(0.0, item_typical_gap)))
        columns["time_decay_itemcf_score"].append(float(time_cf_scores[candidate_index]))
        columns["bpr_score"].append(bpr_value)
        columns["bpr_cosine"].append(bpr_cosine)
        columns["bpr_item_bias"].append(bpr_bias)
        columns["candidate_user_historical_positive_rate"].append(candidate_user_rate)
        columns["candidate_item_historical_positive_rate"].append(candidate_item_rate)
        columns["candidate_pair_active_month_rate"].append(candidate_pair_rate)
        columns["candidate_pair_recent3m_positive_rate"].append(candidate_pair_recent3m_rate)
        columns["candidate_pair_frequency_lift"].append(candidate_pair_lift)
        columns["candidate_user_log_group_size"].append(np.log1p(candidate_group_size))
        columns["candidate_user_prior_bought_fraction"].append(candidate_user_prior_fraction)

    for name in EXTRA_FEATURES:
        base[name] = columns[name]
    return base[COMPETITION_FEATURES].fillna(0)
