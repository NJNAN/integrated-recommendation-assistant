"""Competition-aligned grouped ranking metrics.

The leaderboard evaluates a ranked candidate list independently for every
user.  These helpers therefore never mix scores from different users.  For
temporal replay, ``snapshot_month`` is included in the group key so the same
user in two months is treated as two independent ranking requests.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DEFAULT_WEIGHTS = {"gauc": 0.40, "ndcg@10": 0.40, "recall@10": 0.20}


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def within_group_percentile(
    frame: pd.DataFrame,
    values: Sequence[float] | np.ndarray | pd.Series,
    group_cols: Sequence[str] = ("snapshot_month", "user_id"),
) -> np.ndarray:
    """Return percentile ranks that are comparable only inside each request."""
    result = frame.loc[:, list(group_cols)].copy()
    result["_value"] = np.asarray(values, dtype=float)
    return (
        result.groupby(list(group_cols), sort=False)["_value"]
        .rank(method="average", pct=True)
        .to_numpy(float)
    )


def competition_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    group_cols: Sequence[str] = ("snapshot_month", "user_id"),
    item_col: str = "item_id",
    label_col: str = "label",
    k: int = 10,
) -> dict[str, float | int]:
    """Compute GAUC, NDCG@K, Recall@K and the leaderboard composite.

    GAUC is weighted by candidate-row count, the common production definition.
    ``macro_gauc`` and ``pair_weighted_gauc`` are also returned as sensitivity
    checks because public competition descriptions sometimes leave the exact
    GAUC weighting implicit.  NDCG/Recall average over groups with at least one
    relevant item; zero-positive groups cannot define recall and are reported
    separately.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    required = {*group_cols, item_col, label_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metric frame missing columns: {sorted(missing)}")
    if len(frame) != len(scores):
        raise ValueError("scores must have the same length as frame")

    work = frame.loc[:, [*group_cols, item_col, label_col]].copy()
    work["_score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(work["_score"]).all():
        raise ValueError("scores must all be finite")
    labels = work[label_col].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")

    # Stable item-id tie breaking makes local and submission evaluation exact.
    work[item_col] = work[item_col].astype(str)
    work = work.sort_values(
        [*group_cols, "_score", item_col],
        ascending=[*[True] * len(group_cols), False, True],
        kind="mergesort",
    )
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))

    aucs: list[float] = []
    auc_row_weights: list[int] = []
    auc_pair_weights: list[int] = []
    ndcgs: list[float] = []
    recalls: list[float] = []
    positive_groups = 0
    total_groups = 0
    for _, group in work.groupby(list(group_cols), sort=False, observed=True):
        total_groups += 1
        y = group[label_col].to_numpy(np.int8)
        s = group["_score"].to_numpy(float)
        positives = int(y.sum())
        negatives = len(y) - positives
        auc = _binary_auc(y, s)
        if np.isfinite(auc):
            aucs.append(float(auc))
            auc_row_weights.append(len(y))
            auc_pair_weights.append(positives * negatives)
        if positives == 0:
            continue
        positive_groups += 1
        top = y[:k]
        hits = int(top.sum())
        dcg = float(np.dot(top, discounts[: len(top)]))
        ideal_hits = min(positives, k)
        idcg = float(discounts[:ideal_hits].sum())
        ndcgs.append(dcg / idcg)
        recalls.append(hits / positives)

    if not aucs:
        raise ValueError("GAUC is undefined: no group contains both classes")
    if not ndcgs:
        raise ValueError("NDCG/Recall are undefined: no group has a positive label")

    gauc = float(np.average(aucs, weights=auc_row_weights))
    ndcg = float(np.mean(ndcgs))
    recall = float(np.mean(recalls))
    composite = 0.40 * gauc + 0.40 * ndcg + 0.20 * recall
    return {
        "gauc": gauc,
        "macro_gauc": float(np.mean(aucs)),
        "pair_weighted_gauc": float(np.average(aucs, weights=auc_pair_weights)),
        f"ndcg@{k}": ndcg,
        f"recall@{k}": recall,
        "composite": float(composite),
        "groups": total_groups,
        "positive_groups": positive_groups,
        "auc_groups": len(aucs),
    }


def monthly_competition_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    month_col: str = "snapshot_month",
    k: int = 10,
) -> dict[str, dict[str, float | int]]:
    """Compute the same metric bundle separately for every replay month."""
    values = np.asarray(scores, dtype=float)
    result: dict[str, dict[str, float | int]] = {}
    for month, index in frame.groupby(month_col, sort=True).groups.items():
        positions = frame.index.get_indexer(index)
        part = frame.loc[index]
        result[str(month)] = competition_metrics(part, values[positions], k=k)
    return result
