"""v16 官方评分函数包装（复用项目既有实现，不重新解释指标）。

假设声明
--------
以下 evaluator 是根据项目内已有 `baseline/competition_metrics.py` 的常见
推荐系统定义实现/包装，并非已确认的官方逐行实现。项目内未发现老师提供的
官方逐行 evaluator；本模块与 v15/v21 审计共用同一实现，保证 OOF 选择口径
与历史版本完全一致。

口径要点（详见 baseline.competition_metrics.py）：
- GAUC：按 (snapshot_month, user_id) 分组；主口径按候选行数加权平均；
  同时返回 macro_gauc 与正负对加权 pair_weighted_gauc 作敏感性检查。
- NDCG@10 / Recall@10：只对含正例的组做宏平均；无正例用户不参与（单独计数）。
- tie 处理：score 降序、item_id 升序稳定排序后取 top-K。
- composite = 0.4*GAUC + 0.4*NDCG@10 + 0.2*Recall@10。
"""
from __future__ import annotations

from baseline.competition_metrics import (  # noqa: F401
    competition_metrics as _competition_metrics,
    monthly_competition_metrics as _monthly_competition_metrics,
    within_group_percentile,
)

DEFAULT_GROUP_COLS = ("snapshot_month", "user_id")


def competition_metrics(
    frame,
    score_col: str,
    *,
    group_cols=("snapshot_month", "user_id"),
    k: int = 10,
) -> dict:
    """包装：输入 score 为列名，输出 v16 统一命名。

    frame 必须含 user_id / item_id / label / score_col，以及 group_cols
    （回放 OOF 用 ("snapshot_month","user_id")；最终提交只有一个月时调用方应
    传 ("user_id",)）。
    """
    if score_col not in frame.columns:
        raise ValueError(f"score column {score_col!r} not in frame")
    result = _competition_metrics(
        frame, frame[score_col].to_numpy(float),
        group_cols=group_cols, item_col="item_id", label_col="label", k=k,
    )
    return {
        "gauc": result["gauc"],
        "macro_gauc": result["macro_gauc"],
        "pair_weighted_gauc": result["pair_weighted_gauc"],
        "ndcg10": result[f"ndcg@{k}"],
        "recall10": result[f"recall@{k}"],
        "score": result["composite"],
        "users": int(result["groups"]),
        "positive_users": int(result["positive_groups"]),
        "auc_groups": int(result["auc_groups"]),
    }


def monthly_competition_metrics(frame, score_col: str, *, k: int = 10) -> dict:
    """按月输出同口径指标（键为月份字符串）。"""
    return {
        str(month): competition_metrics(
            part, score_col, group_cols=("snapshot_month", "user_id"), k=k
        )
        for month, part in frame.groupby("snapshot_month", sort=True)
    }
