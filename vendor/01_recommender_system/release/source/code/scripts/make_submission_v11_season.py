"""为老师候选集逐行打分（v11 季节特征版），生成严格三列提交文件。

与 make_submission.py 的关系：协议、校验、meta 记录完全一致，差异只有
“在 competition_features 之后追加 seasonal_features 的 10 列”。模型与
季节状态都从 --models-dir（models/competition_v11_season）读取，不触碰
v5/v8/v9/v10 的任何文件。

用法示例：
    python scripts/make_submission_v11_season.py data/sample_candidates.csv \
        outputs/submissions/submission_traincsv_v11_season_oofblend.csv \
        --blend oof_blend --model-version traincsv-v11-season-oofblend \
        --models-dir models/competition_v11_season

    python scripts/make_submission_v11_season.py data/sample_candidates.csv \
        outputs/submissions/submission_traincsv_v11_season_topkblend.csv \
        --blend topk_blend --model-version traincsv-v11-season-topkblend \
        --models-dir models/competition_v11_season
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.competition import competition_features  # noqa: E402
from baseline.engine import BaselineEngine  # noqa: E402
from baseline.seasonal_features import add_seasonal_columns  # noqa: E402

DEFAULT_MODELS = ROOT / "models" / "competition_v11_season"
ALIASES = {
    ("user_id", "item_id"): ("user_id", "item_id"),
    ("CustomerID", "StockCode"): ("CustomerID", "StockCode"),
}
DIRECT_FEATURES = {
    "bpr": "bpr_score",
    "timecf": "time_decay_itemcf_score",
    "hazard": "user_item_next30_cycle_score",
    "pairrate": "candidate_pair_recent3m_positive_rate",
}
MODEL_ALIASES = {"lgb": "lightgbm", "lr": "lr", "cat": "catboost"}
SINGLE_CHOICES = ("lightgbm", "lr", "catboost")
BLEND_CHOICES = ("oof_blend", "topk_blend")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    selected = next((pair for pair in ALIASES if set(pair).issubset(frame.columns)), None)
    if selected is None:
        raise ValueError("候选集必须包含 user_id,item_id 或 CustomerID,StockCode")
    if len(frame.columns) != 2:
        raise ValueError(f"候选集应恰好 2 列，实际为：{list(frame.columns)}")
    frame = frame.loc[:, list(selected)].rename(
        columns={selected[0]: "user_id", selected[1]: "item_id"}
    )
    frame["user_id"] = frame["user_id"].str.strip()
    frame["item_id"] = frame["item_id"].str.strip()
    empty = (frame["user_id"] == "") | (frame["item_id"] == "")
    if empty.any():
        rows = (np.flatnonzero(empty.to_numpy()) + 2).tolist()[:10]
        raise ValueError(f"候选集存在空 ID，CSV 行号示例：{rows}")
    return frame


def model_input(ranker, features: pd.DataFrame, uid: str, item_ids: list[str]) -> pd.DataFrame:
    """选择与训练时完全一致的列（含季节特征；兼容 CatBoost 的 user/item 类别列）。"""
    expanded = features.copy()
    expanded.insert(0, "item_id", item_ids)
    expanded.insert(0, "user_id", [str(uid)] * len(expanded))
    raw_names = getattr(ranker, "feature_names_in_", None)
    names = list(raw_names) if raw_names is not None else []
    if not names:
        raw_names = getattr(ranker, "feature_names_", None)
        names = list(raw_names) if raw_names is not None else []
    if not names and hasattr(ranker, "booster_"):
        names = list(ranker.booster_.feature_name())
    if not names:
        raise ValueError(f"模型 {type(ranker).__name__} 无法读取训练特征名")
    missing = [name for name in names if name not in expanded]
    if missing:
        raise ValueError(f"模型需要但当前无法生成的特征：{missing}")
    return expanded.loc[:, names]


def score_components(
    frame: pd.DataFrame,
    engine: BaselineEngine,
    rankers: dict,
    seasonal_state: dict,
    component_names: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """单次遍历生成全部组件分数（模型 + 直接特征），顺序与行数严格保持。"""
    values = {name: np.zeros(len(frame), dtype=float) for name in component_names}
    unknown_items = 0
    for uid, group in frame.groupby("user_id", sort=False):
        known_mask = group["item_id"].isin(engine.pos)
        unknown_items += int((~known_mask).sum())
        known = group.loc[known_mask]
        if known.empty:
            continue
        recalled, raw_scores, _ = engine.candidates(str(uid))
        source_map = {row["item_id"]: row["sources"] for row in recalled}
        candidates = [
            {"item_id": item, "sources": source_map.get(item, [])}
            for item in known["item_id"].tolist()
        ]
        features = competition_features(engine, str(uid), candidates, raw_scores)
        seasonal = add_seasonal_columns(
            features, seasonal_state, str(uid), known["item_id"].tolist()
        )
        features = pd.concat([features, seasonal], axis=1)
        indexes = known.index.to_numpy()
        for name in component_names:
            if name in DIRECT_FEATURES:
                values[name][indexes] = features[DIRECT_FEATURES[name]].to_numpy(float)
                continue
            model_name = MODEL_ALIASES.get(name, name)
            if model_name not in rankers:
                raise ValueError(f"OOF 融合需要模型 {name!r}，但 rankers.pkl 中不存在")
            ranker = rankers[model_name]
            model_frame = model_input(ranker, features, str(uid), known["item_id"].tolist())
            if hasattr(ranker, "predict_proba"):
                prediction = ranker.predict_proba(model_frame)[:, 1]
            else:
                prediction = ranker.predict(model_frame)
            values[name][indexes] = np.asarray(prediction, dtype=float)
    return values, unknown_items


def blend_scores(
    candidates: pd.DataFrame,
    component_values: dict[str, np.ndarray],
    blend_spec: dict,
) -> np.ndarray:
    scope = blend_spec.get("rank_scope", "global")
    global_weight = float(blend_spec.get("global_rank_weight", 1.0))
    if scope not in {"global", "per_user", "hybrid_global_user"}:
        raise ValueError(f"未知 OOF 融合秩范围：{scope!r}")
    if not 0.0 <= global_weight <= 1.0:
        raise ValueError("global_rank_weight 必须位于 [0, 1]")
    values = np.zeros(len(candidates), dtype=float)
    for name, weight in blend_spec.get("components", {}).items():
        series = pd.Series(component_values[name], index=candidates.index)
        global_rank = series.rank(method="average", pct=True).to_numpy()
        if scope == "global":
            ranked = global_rank
        else:
            rank_frame = candidates[["user_id"]].copy()
            rank_frame["score"] = series.to_numpy()
            user_rank = rank_frame.groupby("user_id", sort=False)["score"].rank(
                method="average", pct=True
            ).to_numpy()
            ranked = (
                global_weight * global_rank + (1.0 - global_weight) * user_rank
                if scope == "hybrid_global_user"
                else user_rank
            )
        values += float(weight) * ranked
    return values


def validate_submission(candidates: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["user_id", "item_id", "score"]:
        raise ValueError("提交文件列名或顺序错误")
    if len(submission) != len(candidates):
        raise ValueError("提交行数与候选集不一致")
    if not submission[["user_id", "item_id"]].equals(candidates[["user_id", "item_id"]]):
        raise ValueError("提交文件未逐行覆盖原候选集")
    if not submission["score"].map(lambda value: math.isfinite(float(value))).all():
        raise ValueError("score 必须全部为有限数值")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v11 季节特征提交生成")
    parser.add_argument("candidate_csv", type=Path, help="老师下发的两列候选 CSV")
    parser.add_argument("output_csv", type=Path, help="输出提交 CSV")
    parser.add_argument("--model", default="oof_blend",
                        choices=[*SINGLE_CHOICES, *BLEND_CHOICES, "spec"],
                        help="精排模型或融合版本")
    parser.add_argument("--model-version", required=True, help="排行榜中填写的模型版本号")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--spec-json", type=Path, default=None,
                        help="--model spec 时使用的外部融合规格 JSON（含 components/rank_scope）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_version.strip():
        raise ValueError("--model-version 不能为空")
    if args.candidate_csv.resolve() == args.output_csv.resolve():
        raise ValueError("输出路径不能覆盖老师下发的候选文件")

    candidates = read_candidates(args.candidate_csv)
    model_meta_path = args.models_dir / "model_meta.json"
    if not model_meta_path.exists():
        raise ValueError(f"缺少模型元数据：{model_meta_path}")
    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))

    expected = model_meta.get("submission_versions", {}).get(args.model)
    if expected and args.model_version.strip() != expected:
        raise ValueError(
            f"模型版本号不匹配：{args.model!r} 应填写 {expected!r}，"
            f"当前为 {args.model_version.strip()!r}"
        )

    external_spec = None
    if args.model == "spec":
        if args.spec_json is None or not args.spec_json.exists():
            raise ValueError("--model spec 需要存在的 --spec-json")
        external_spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
        expected = external_spec.get("model_version")
        if expected and args.model_version.strip() != expected:
            raise ValueError(
                f"spec 版本号不匹配：spec 声明 {expected!r}，"
                f"当前为 {args.model_version.strip()!r}"
            )

    seasonal_path = args.models_dir / "seasonal_state.pkl"
    if not seasonal_path.exists():
        raise ValueError(f"缺少季节状态：{seasonal_path}")
    with seasonal_path.open("rb") as stream:
        seasonal_state = pickle.load(stream)
    engine = BaselineEngine.load(args.models_dir)
    with (args.models_dir / "rankers.pkl").open("rb") as stream:
        rankers = pickle.load(stream)

    if args.model in SINGLE_CHOICES:
        if args.model not in rankers:
            raise ValueError(f"模型 {args.model!r} 不存在；可用：{sorted(rankers)}")
        component_values, unknown_items = score_components(
            candidates, engine, rankers, seasonal_state, [args.model]
        )
        result = component_values[args.model]
    elif args.model in BLEND_CHOICES:
        key = "blend_spec" if args.model == "oof_blend" else "blend_spec_topk"
        blend_spec = model_meta.get(key, {})
        component_weights = blend_spec.get("components", {})
        if not component_weights:
            raise ValueError(f"model_meta.json 缺少 {key} 融合权重")
        component_values, unknown_items = score_components(
            candidates, engine, rankers, seasonal_state, list(component_weights)
        )
        result = blend_scores(candidates, component_values, blend_spec)
    elif args.model == "spec":
        blend_spec = external_spec
        component_weights = blend_spec.get("components", {})
        if not component_weights:
            raise ValueError("spec 缺少 components 融合权重")
        component_values, unknown_items = score_components(
            candidates, engine, rankers, seasonal_state, list(component_weights)
        )
        result = blend_scores(candidates, component_values, blend_spec)
    else:
        raise ValueError(f"未知模型选择：{args.model}")

    known = candidates["item_id"].isin(engine.pos).to_numpy()
    result[~known] = 0.0
    submission = candidates.copy()
    submission["score"] = result
    validate_submission(candidates, submission)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False, encoding="utf-8")
    meta_path = args.output_csv.with_suffix(".meta.json")
    recorded_spec = None
    if args.model in {"oof_blend", "topk_blend"}:
        recorded_spec = model_meta.get(
            "blend_spec" if args.model == "oof_blend" else "blend_spec_topk"
        )
    elif args.model == "spec":
        recorded_spec = external_spec
    metadata = {
        "model_version": args.model_version.strip(),
        "ranker": args.model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(args.candidate_csv.resolve()),
        "candidate_sha256": sha256(args.candidate_csv),
        "rankers_sha256": sha256(args.models_dir / "rankers.pkl"),
        "model_meta_sha256": sha256(model_meta_path),
        "seasonal_state_sha256": sha256(seasonal_path),
        "training_sha256": model_meta.get("training_sha256"),
        "blend_spec": recorded_spec,
        "rows": len(submission),
        "unique_users": int(submission["user_id"].nunique()),
        "unique_items": int(submission["item_id"].nunique()),
        "unknown_item_rows_scored_zero": unknown_items,
        "csv_columns": list(submission.columns),
        "coverage_check": "passed",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"提交文件：{args.output_csv} ({len(submission)} 行)")
    print(f"版本记录：{meta_path}")
    print(f"排行榜模型版本号：{metadata['model_version']}")
    if unknown_items:
        print(f"警告：{unknown_items} 行商品不在训练商品表中，已保留并赋 0 分")


if __name__ == "__main__":
    main()
