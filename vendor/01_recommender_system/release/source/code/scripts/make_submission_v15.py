"""生成 v15 活跃度门控双专家与 v11 主模型的校准融合提交。"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.competition import competition_features  # noqa: E402
from baseline.engine import BaselineEngine  # noqa: E402
from baseline.features_v15 import append_v15_interactions, user_activity_frame  # noqa: E402
from baseline.seasonal_features import add_seasonal_columns  # noqa: E402
from scripts.make_submission_v11_season import (  # noqa: E402
    blend_scores,
    read_candidates,
    score_components,
    sha256,
    validate_submission,
)
from scripts.train_competition_v15 import calibrated_predict  # noqa: E402

DEFAULT_V11_MODELS = ROOT / "models" / "competition_v11_season"
DEFAULT_V15_MODELS = ROOT / "models" / "competition_v15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v15 分解式融合提交生成")
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--v11-models-dir", type=Path, default=DEFAULT_V11_MODELS)
    parser.add_argument("--v15-models-dir", type=Path, default=DEFAULT_V15_MODELS)
    return parser.parse_args()


def complete_feature_frame(candidates: pd.DataFrame, engine, seasonal_state) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    known_all = candidates.item_id.isin(engine.pos).to_numpy()
    for uid, group in candidates.groupby("user_id", sort=False):
        known = group.loc[group.item_id.isin(engine.pos)]
        if known.empty:
            continue
        recalled, raw_scores, _ = engine.candidates(str(uid))
        source_map = {row["item_id"]: row["sources"] for row in recalled}
        item_ids = known.item_id.astype(str).tolist()
        rows = [{"item_id": item, "sources": source_map.get(item, [])} for item in item_ids]
        base = competition_features(engine, str(uid), rows, raw_scores)
        season = add_seasonal_columns(base, seasonal_state, str(uid), item_ids)
        full = append_v15_interactions(pd.concat([base, season], axis=1))
        full.insert(0, "item_id", item_ids)
        full.insert(0, "user_id", [str(uid)] * len(full))
        full.insert(0, "snapshot_month", ["2010-11"] * len(full))
        full.insert(0, "_row", known.index.to_numpy())
        frames.append(full)
    result = pd.concat(frames, ignore_index=True).sort_values("_row").reset_index(drop=True)
    expected = np.flatnonzero(known_all)
    if not np.array_equal(result._row.to_numpy(int), expected):
        raise ValueError("v15 特征行与候选集已知商品行未一一对齐")
    return result, known_all


def v15_scores(features: pd.DataFrame, package: dict) -> np.ndarray:
    activity_features = package["activity_features"]
    item_features = package["features"]
    users = user_activity_frame(features)
    pa_user = package["activity"].predict_proba(users[activity_features])[:, 1]
    users["p_activity"] = pa_user
    pa = features[["snapshot_month", "user_id"]].merge(
        users[["snapshot_month", "user_id", "p_activity"]],
        on=["snapshot_month", "user_id"], how="left", validate="many_to_one",
    ).p_activity.to_numpy(float)
    prior = features.prior_bought.round().astype(int).to_numpy()
    pc = np.zeros(len(features), dtype=float)
    for flag, name in ((0, "first_purchase"), (1, "repurchase")):
        mask = prior == flag
        pc[mask] = package["experts"][name].predict_proba(features.loc[mask, item_features])[:, 1]
    return calibrated_predict(package["calibrator"], features, pa, pc)


def pct_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, float), method="average") / len(values)


def main() -> None:
    args = parse_args()
    if not args.model_version.strip():
        raise ValueError("--model-version 不能为空")
    if args.candidate_csv.resolve() == args.output_csv.resolve():
        raise ValueError("输出不能覆盖候选集")
    candidates = read_candidates(args.candidate_csv)

    v11_meta_path = args.v11_models_dir / "model_meta.json"
    v15_meta_path = args.v15_models_dir / "model_meta.json"
    v11_meta = json.loads(v11_meta_path.read_text(encoding="utf-8"))
    v15_meta = json.loads(v15_meta_path.read_text(encoding="utf-8"))
    expected_version = v15_meta["submission_version"]
    if args.model_version.strip() != expected_version:
        raise ValueError(f"模型版本号应为 {expected_version!r}")

    with (args.v11_models_dir / "seasonal_state.pkl").open("rb") as stream:
        seasonal_state = pickle.load(stream)
    with (args.v11_models_dir / "rankers.pkl").open("rb") as stream:
        rankers = pickle.load(stream)
    with (args.v15_models_dir / "v15_models.pkl").open("rb") as stream:
        package = pickle.load(stream)
    engine = BaselineEngine.load(args.v11_models_dir)

    blend_spec = v11_meta["blend_spec"]
    components, unknown = score_components(
        candidates, engine, rankers, seasonal_state, list(blend_spec["components"])
    )
    v11 = blend_scores(candidates, components, blend_spec)
    full, known = complete_feature_frame(candidates, engine, seasonal_state)
    raw_v15_known = v15_scores(full, package)
    raw_v15 = np.zeros(len(candidates), dtype=float)
    raw_v15[known] = raw_v15_known
    v11[~known] = 0.0

    weight = float(v15_meta["blend_spec"]["v15_weight"])
    score = weight * pct_rank(raw_v15) + (1.0 - weight) * pct_rank(v11)
    score[~known] = 0.0
    submission = candidates.copy()
    submission["score"] = score
    validate_submission(candidates, submission)
    if not all(math.isfinite(float(value)) for value in submission.score):
        raise ValueError("提交包含非有限分数")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False, encoding="utf-8")
    metadata = {
        "model_version": args.model_version.strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(args.candidate_csv.resolve()),
        "candidate_sha256": sha256(args.candidate_csv),
        "v11_model_meta_sha256": sha256(v11_meta_path),
        "v15_model_meta_sha256": sha256(v15_meta_path),
        "v15_models_sha256": sha256(args.v15_models_dir / "v15_models.pkl"),
        "blend_spec": v15_meta["blend_spec"],
        "rows": len(submission),
        "unique_users": int(submission.user_id.nunique()),
        "unique_items": int(submission.item_id.nunique()),
        "unknown_item_rows_scored_zero": int((~known).sum()),
        "feature_alignment_rows": len(full),
        "csv_columns": list(submission.columns),
        "coverage_check": "passed",
    }
    meta_path = args.output_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
