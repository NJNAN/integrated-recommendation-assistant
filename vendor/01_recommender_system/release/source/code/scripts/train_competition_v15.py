"""v15：用户活跃度门控 + 复购/首购双专家 + 严格前向校准。

验证仍使用 v9 的 2010-06..10 固定候选回放，但所有季节特征只在 2010-06
之后使用，因此不会继承 v13 在 1/2 月追加 2010-01/02 聚合的时间穿越问题。
外层验证为 08/09/10；每一折的校准器只在更早的内层月份拟合。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.competition import COMPETITION_FEATURES  # noqa: E402
from baseline.features_v15 import (  # noqa: E402
    INTERACTION_FEATURES_V15,
    USER_ACTIVITY_FEATURES,
    append_v15_interactions,
    user_activity_frame,
)
from baseline.seasonal_features import (  # noqa: E402
    SEASONAL_FEATURES,
    append_seasonal_frame,
    build_seasonal_state,
)
from scripts.train_competition import file_sha256, load_training_data  # noqa: E402
from scripts.train_competition_v9 import fast_auc, temporal_weights  # noqa: E402

DEFAULT_SAMPLES = ROOT / "outputs" / "competition_v9" / "candidate_aligned_samples.csv"
DEFAULT_V11_OOF = ROOT / "outputs" / "competition_v11_season" / "candidate_aligned_oof.csv"
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "sample_candidates.csv"
DEFAULT_MODELS = ROOT / "models" / "competition_v15"
DEFAULT_OUTPUTS = ROOT / "outputs" / "competition_v15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v15 分解式购买概率模型")
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--v11-oof-csv", type=Path, default=DEFAULT_V11_OOF)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--model-version", default="traincsv-v15-activity-moe")
    parser.add_argument("--temporal-decay", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_lgbm(seed: int, role: str):
    from lightgbm import LGBMClassifier

    if role == "activity":
        params = dict(n_estimators=260, num_leaves=23, learning_rate=0.025,
                      min_child_samples=35, reg_alpha=0.3, reg_lambda=5.0)
    else:
        params = dict(n_estimators=420, num_leaves=31, learning_rate=0.025,
                      min_child_samples=70, reg_alpha=0.3, reg_lambda=6.0)
    return LGBMClassifier(
        objective="binary", random_state=seed, n_jobs=8, verbosity=-1,
        deterministic=True, force_col_wise=True, max_bin=255,
        colsample_bytree=0.9, subsample=0.9, subsample_freq=1, **params,
    )


def add_activity_labels(samples: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    transactions = clean[["user_id", "transaction_time"]].copy()
    transactions["snapshot_month"] = transactions.transaction_time.dt.to_period("M").astype(str)
    active = set(zip(transactions.snapshot_month, transactions.user_id.astype(str)))
    keys = list(zip(samples.snapshot_month.astype(str), samples.user_id.astype(str)))
    result = samples.copy()
    result["activity_label"] = np.fromiter((int(key in active) for key in keys), int, len(keys))
    return result


def row_weights(frame: pd.DataFrame, latest: str, decay: float, positive_weight: float) -> np.ndarray:
    weights = temporal_weights(frame, latest, decay, positive_weight=1.0)
    if "label" in frame:
        weights *= np.where(frame.label.to_numpy(int) == 1, positive_weight, 1.0)
    return weights


def fit_base(train: pd.DataFrame, features: list[str], seed: int, decay: float) -> dict:
    users = user_activity_frame(train)
    activity_targets = train.groupby(["snapshot_month", "user_id"], sort=False).activity_label.first()
    users = users.merge(activity_targets.rename("activity_label").reset_index(),
                        on=["snapshot_month", "user_id"], validate="one_to_one")
    activity = make_lgbm(seed, "activity")
    activity.fit(
        users[USER_ACTIVITY_FEATURES], users.activity_label,
        sample_weight=row_weights(users.rename(columns={"activity_label": "label"}),
                                  users.snapshot_month.max(), decay, 1.0),
    )

    conditional = train.loc[train.activity_label == 1].copy()
    experts = {}
    for flag, name in ((0, "first_purchase"), (1, "repurchase")):
        subset = conditional.loc[conditional.prior_bought.round().astype(int) == flag]
        if subset.label.nunique() != 2:
            raise ValueError(f"{name} 专家训练集缺少正例或负例")
        model = make_lgbm(seed + 11 + flag, "expert")
        model.fit(
            subset[features], subset.label,
            sample_weight=row_weights(subset, subset.snapshot_month.max(), decay, 2.0),
        )
        experts[name] = model
    return {"activity": activity, "experts": experts}


def predict_base(models: dict, frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    user_table = user_activity_frame(frame)
    pa = models["activity"].predict_proba(user_table[USER_ACTIVITY_FEATURES])[:, 1]
    user_table["p_activity"] = pa
    mapped = frame[["snapshot_month", "user_id"]].merge(
        user_table[["snapshot_month", "user_id", "p_activity"]],
        on=["snapshot_month", "user_id"], how="left", validate="many_to_one",
    )["p_activity"].to_numpy(float)

    pc = np.zeros(len(frame), dtype=float)
    prior = frame.prior_bought.round().astype(int).to_numpy()
    for flag, name in ((0, "first_purchase"), (1, "repurchase")):
        mask = prior == flag
        pc[mask] = models["experts"][name].predict_proba(frame.loc[mask, features])[:, 1]
    return mapped, pc


def calibration_matrix(pa: np.ndarray, pc: np.ndarray, prior: np.ndarray) -> np.ndarray:
    eps = 1e-5
    la = logit(np.clip(pa, eps, 1 - eps))
    lc = logit(np.clip(pc, eps, 1 - eps))
    prior = np.asarray(prior, dtype=float)
    return np.column_stack([la, lc, prior, lc * prior])


def fit_calibrator(frame: pd.DataFrame, pa: np.ndarray, pc: np.ndarray, seed: int):
    model = LogisticRegression(C=0.35, max_iter=1000, random_state=seed)
    model.fit(calibration_matrix(pa, pc, frame.prior_bought.to_numpy()), frame.label)
    return model


def calibrated_predict(calibrator, frame: pd.DataFrame, pa: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(
        calibration_matrix(pa, pc, frame.prior_bought.to_numpy())
    )[:, 1]


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, float), method="average") / len(values)


def monthly_metrics(frame: pd.DataFrame, score: str) -> dict:
    monthly = {}
    for month, group in frame.groupby("snapshot_month", sort=True):
        monthly[str(month)] = float(roc_auc_score(group.label, group[score]))
    return {
        "pooled_auc": float(roc_auc_score(frame.label, frame[score])),
        "monthly_auc": monthly,
        "mean_monthly_auc": float(np.mean(list(monthly.values()))),
        "worst_month_auc": float(np.min(list(monthly.values()))),
    }


def nested_forward_oof(samples: pd.DataFrame, features: list[str], seed: int, decay: float) -> pd.DataFrame:
    months = sorted(samples.snapshot_month.unique())
    folds = []
    for fold_index, valid_month in enumerate(months[2:]):
        before = [month for month in months if month < valid_month]
        calibration_month = before[-1]
        inner_train = samples.loc[samples.snapshot_month < calibration_month]
        calibration = samples.loc[samples.snapshot_month == calibration_month]
        valid = samples.loc[samples.snapshot_month == valid_month]
        print(f"v15 {valid_month}: inner_train<{calibration_month}, calibrate={calibration_month}, "
              f"outer_train={len(samples.loc[samples.snapshot_month < valid_month]):,}, "
              f"valid={len(valid):,}", flush=True)

        inner_models = fit_base(inner_train, features, seed + fold_index * 20, decay)
        pa_cal, pc_cal = predict_base(inner_models, calibration, features)
        calibrator = fit_calibrator(calibration, pa_cal, pc_cal, seed + 100 + fold_index)

        outer_train = samples.loc[samples.snapshot_month < valid_month]
        outer_models = fit_base(outer_train, features, seed + 200 + fold_index * 20, decay)
        pa, pc = predict_base(outer_models, valid, features)
        fold = valid[["user_id", "item_id", "label", "snapshot_month", "prior_bought"]].copy()
        fold["p_activity"] = pa
        fold["p_item_given_active"] = pc
        fold["decomposed_product"] = pa * pc
        fold["v15_calibrated"] = calibrated_predict(calibrator, valid, pa, pc)
        folds.append(fold)
    return pd.concat(folds, ignore_index=True)


def align_baseline(oof: pd.DataFrame, path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(path, usecols=["user_id", "item_id", "label", "snapshot_month", "blend"],
                           dtype={"user_id": str, "item_id": str, "snapshot_month": str})
    keys = ["user_id", "item_id", "label", "snapshot_month"]
    if baseline.duplicated(keys).any() or oof.duplicated(keys).any():
        raise ValueError("OOF 对齐键不唯一")
    merged = oof.merge(baseline.rename(columns={"blend": "v11_blend"}),
                       on=keys, how="left", validate="one_to_one")
    if merged.v11_blend.isna().any() or len(merged) != len(oof):
        raise ValueError("v15 与 v11 OOF 配对无法一一对齐")
    return merged


def select_blend(oof: pd.DataFrame) -> dict:
    ranked = oof.copy()
    ranked["v15_rank"] = ranked.groupby("snapshot_month")["v15_calibrated"].rank(pct=True)
    ranked["v11_rank"] = ranked.groupby("snapshot_month")["v11_blend"].rank(pct=True)
    records = []
    for weight in np.linspace(0, 1, 41):
        ranked["candidate"] = weight * ranked.v15_rank + (1 - weight) * ranked.v11_rank
        metrics = monthly_metrics(ranked, "candidate")
        objective = 0.65 * metrics["mean_monthly_auc"] + 0.35 * metrics["worst_month_auc"]
        records.append((objective, metrics["pooled_auc"], float(weight), metrics))
    _, _, weight, metrics = max(records)
    return {"v15_weight": weight, "v11_weight": 1.0 - weight,
            "selection_objective": "0.65*mean monthly AUC + 0.35*worst-month AUC",
            "oof_metrics": metrics}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.samples_csv,
                          dtype={"user_id": str, "item_id": str, "snapshot_month": str})
    if args.smoke:
        samples = samples.loc[samples.snapshot_month.isin(sorted(samples.snapshot_month.unique())[:3])]
    clean, _ = load_training_data(args.train_csv)
    samples = append_seasonal_frame(samples, build_seasonal_state(clean))
    samples = append_v15_interactions(samples)
    samples = add_activity_labels(samples, clean)
    features = [*COMPETITION_FEATURES, *SEASONAL_FEATURES, *INTERACTION_FEATURES_V15]
    missing = set(features) - set(samples.columns)
    if missing:
        raise ValueError(f"v15 样本缺少列：{sorted(missing)}")

    # 真实活动标签与“候选中至少一条正例”不是同一个概念，显式记录差异。
    user_check = samples.groupby(["snapshot_month", "user_id"], sort=False).agg(
        activity_label=("activity_label", "first"), candidate_hit=("label", "max")
    )
    print("用户月活跃率:", float(user_check.activity_label.mean()),
          "候选命中率:", float(user_check.candidate_hit.mean()), flush=True)

    if args.smoke:
        # 三个月时外层只有最后一个月，依然可验证全链路。
        original = sorted(samples.snapshot_month.unique())
        if len(original) < 3:
            raise ValueError("smoke 至少需要 3 个月")
    oof = nested_forward_oof(samples, features, args.seed, args.temporal_decay)
    oof = align_baseline(oof, args.v11_oof_csv)
    metrics = {
        "v11_blend": monthly_metrics(oof, "v11_blend"),
        "decomposed_product": monthly_metrics(oof, "decomposed_product"),
        "v15_calibrated": monthly_metrics(oof, "v15_calibrated"),
    }
    blend_spec = select_blend(oof)
    oof["v15_rank"] = oof.groupby("snapshot_month")["v15_calibrated"].rank(pct=True)
    oof["v11_rank"] = oof.groupby("snapshot_month")["v11_blend"].rank(pct=True)
    oof["blend_v15"] = (blend_spec["v15_weight"] * oof.v15_rank
                        + blend_spec["v11_weight"] * oof.v11_rank)
    metrics["blend_v15"] = monthly_metrics(oof, "blend_v15")
    oof.to_csv(args.outputs_dir / "candidate_aligned_oof_v15.csv", index=False)

    final_models = fit_base(samples, features, args.seed + 700, args.temporal_decay)
    # 最终校准器只用真正的外层 OOF 预测，杜绝用同一行标签校准自身预测。
    final_calibrator = fit_calibrator(oof, oof.p_activity.to_numpy(),
                                      oof.p_item_given_active.to_numpy(), args.seed + 900)
    package = {**final_models, "calibrator": final_calibrator,
               "features": features, "activity_features": USER_ACTIVITY_FEATURES}
    with (args.models_dir / "v15_models.pkl").open("wb") as stream:
        pickle.dump(package, stream)

    meta = {
        "model_family_version": args.model_version,
        "submission_version": f"{args.model_version}-calibrated-blend",
        "training_sha256": file_sha256(args.train_csv),
        "candidate_sha256": file_sha256(args.candidate_csv),
        "source_samples": str(args.samples_csv.resolve()),
        "validation_protocol": "nested forward: inner previous-month calibration; outer 2010-08/09/10",
        "seasonal_boundary": "v15 trains on 2010-06..10 only; static state ending 2010-03 is available at every snapshot",
        "activity_target": "user has any transaction in target month (not merely a positive candidate row)",
        "experts": ["first_purchase(prior_bought=0)", "repurchase(prior_bought=1)"],
        "features": features,
        "metrics": metrics,
        "blend_spec": blend_spec,
        "user_month_activity_rate": float(user_check.activity_label.mean()),
        "user_month_candidate_hit_rate": float(user_check.candidate_hit.mean()),
        "seconds": time.perf_counter() - started,
    }
    (args.models_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
