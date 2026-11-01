"""Candidate-aligned temporal replay for the teacher's fixed scoring rows.

Unlike the earlier 1:5 random-negative training set, each monthly snapshot here
scores the same user-item pairs that appear in ``sample_candidates.csv`` and
labels them from that month's observed purchases.  Candidate membership remains
unlabelled; all labels still come exclusively from ``train.csv``.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.bpr import attach_bpr_state, train_time_decay_bpr  # noqa: E402
from baseline.competition import (  # noqa: E402
    COMPETITION_FEATURES,
    attach_competition_state,
    competition_features,
)
from baseline.config import Config  # noqa: E402
from baseline.data import sequences  # noqa: E402
from baseline.engine import BaselineEngine, train_item2vec  # noqa: E402
from scripts.train_competition import file_sha256, load_training_data  # noqa: E402

DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "sample_candidates.csv"
DEFAULT_SOURCE_MODELS = ROOT / "models" / "competition_v8"
DEFAULT_MODELS = ROOT / "models" / "competition_v9"
DEFAULT_OUTPUTS = ROOT / "outputs" / "competition_v9"


LGBM_CONFIGS = {
    "deep47_w1": ({
        "n_estimators": 420, "num_leaves": 47, "learning_rate": 0.025,
        "min_child_samples": 65, "reg_alpha": 0.25, "reg_lambda": 5.0,
        "colsample_bytree": 0.88, "subsample": 0.90, "subsample_freq": 1,
    }, 1.0),
    "deep47_w3": ({
        "n_estimators": 420, "num_leaves": 47, "learning_rate": 0.025,
        "min_child_samples": 65, "reg_alpha": 0.25, "reg_lambda": 5.0,
        "colsample_bytree": 0.88, "subsample": 0.90, "subsample_freq": 1,
    }, 3.0),
    "regularized31_w2": ({
        "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.022,
        "min_child_samples": 85, "reg_alpha": 0.35, "reg_lambda": 6.0,
        "colsample_bytree": 0.92, "subsample": 0.92, "subsample_freq": 1,
    }, 2.0),
    "shallow15_w3": ({
        "n_estimators": 560, "num_leaves": 15, "learning_rate": 0.020,
        "min_child_samples": 80, "reg_alpha": 0.25, "reg_lambda": 4.0,
        "colsample_bytree": 0.94, "subsample": 0.92, "subsample_freq": 1,
    }, 3.0),
    "depth6_w5": ({
        "n_estimators": 480, "num_leaves": 23, "max_depth": 6,
        "learning_rate": 0.024, "min_child_samples": 70,
        "reg_alpha": 0.25, "reg_lambda": 5.0,
        "colsample_bytree": 0.90, "subsample": 0.90, "subsample_freq": 1,
    }, 5.0),
}

DIRECT_COMPONENTS = {
    "bpr": "bpr_score",
    "timecf": "time_decay_itemcf_score",
    "hazard": "user_item_next30_cycle_score",
    "pairrate": "candidate_pair_recent3m_positive_rate",
}


def make_lgbm(seed: int, parameters: dict):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary", random_state=seed, n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True, max_bin=255, **parameters,
    )


def make_lr(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1400, random_state=seed),
    )


def make_catboost(seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=520, depth=7, learning_rate=0.035,
        loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=6.0,
        random_strength=0.35, random_seed=seed, thread_count=4,
        verbose=False, allow_writing_files=False,
    )


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average") / len(values)


def temporal_weights(frame: pd.DataFrame, latest: str, decay: float, positive_weight: float = 1.0) -> np.ndarray:
    latest_period = pd.Period(latest, freq="M")
    distance = np.asarray([
        latest_period.ordinal - pd.Period(value, freq="M").ordinal
        for value in frame.snapshot_month
    ])
    weights = np.power(decay, np.maximum(distance, 0)).astype(float)
    weights *= np.where(frame.label.to_numpy(int) == 1, positive_weight, 1.0)
    return weights


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def metric_bundle(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    global_auc = fast_auc(frame.label.to_numpy(), scores)
    grouped = frame.assign(_score=scores).groupby(["snapshot_month", "user_id"], sort=False)
    aucs, sizes = [], []
    for _, group in grouped:
        auc = fast_auc(group.label.to_numpy(), group._score.to_numpy())
        if np.isfinite(auc):
            aucs.append(auc)
            sizes.append(len(group))
    weighted_gauc = float(np.average(aucs, weights=sizes))
    macro_gauc = float(np.mean(aucs))
    robust = 0.55 * global_auc + 0.30 * weighted_gauc + 0.15 * macro_gauc
    return {
        "auc": global_auc, "weighted_gauc": weighted_gauc,
        "macro_gauc": macro_gauc, "robust_score": robust,
    }


def build_aligned_sample(
    engine: BaselineEngine,
    candidates: pd.DataFrame,
    future: pd.DataFrame,
    month: str,
) -> pd.DataFrame:
    positive_pairs = set(zip(future.user_id.astype(str), future.item_id.astype(str)))
    rows = []
    for number, (uid, group) in enumerate(candidates.groupby("user_id", sort=False)):
        uid = str(uid)
        if uid not in engine.histories:
            continue
        known = group.loc[group.item_id.isin(engine.pos)]
        if known.empty:
            continue
        recalled, scores, _ = engine.candidates(uid)
        source_map = {row["item_id"]: row["sources"] for row in recalled}
        item_ids = known.item_id.astype(str).tolist()
        candidate_rows = [
            {"item_id": item, "sources": source_map.get(item, [])} for item in item_ids
        ]
        features = competition_features(engine, uid, candidate_rows, scores)
        features.insert(0, "item_id", item_ids)
        features.insert(0, "user_id", uid)
        features["label"] = [int((uid, item) in positive_pairs) for item in item_ids]
        features["snapshot_month"] = month
        rows.append(features)
        if number and number % 300 == 0:
            print(f"候选回放用户：{number}", flush=True)
    result = pd.concat(rows, ignore_index=True)
    if result.label.nunique() != 2:
        raise ValueError(f"{month} 候选回放样本缺少正例或负例")
    return result


def build_snapshot(
    clean: pd.DataFrame,
    catalog: pd.DataFrame,
    order_times: pd.Series,
    candidates: pd.DataFrame,
    item_weights: dict[str, float],
    month: pd.Period,
    cfg: Config,
    bpr_epochs: int,
) -> pd.DataFrame:
    start, end = month.start_time, (month + 1).start_time
    history = clean.loc[order_times < start].copy()
    future = clean.loc[(order_times >= start) & (order_times < end)].copy()
    snapshot_cfg = replace(cfg, history_end=str(start.date()), test_start=str(end.date()), eval_users=0)
    print(f"\nv9 候选回放 {month}: 历史 {len(history):,}，标签 {len(future):,}", flush=True)
    w2v = train_item2vec(sequences(history), snapshot_cfg)
    engine = BaselineEngine(history, catalog, snapshot_cfg, w2v)
    attach_competition_state(engine, history, item_weights, candidate_frame=candidates)
    bpr = train_time_decay_bpr(
        history, item_weights, seed=cfg.seed + month.ordinal, epochs=bpr_epochs,
    )
    attach_bpr_state(engine, bpr)
    sample = build_aligned_sample(engine, candidates, future, str(month))
    print(
        f"v9 {month}: {len(sample):,} 行，正例 {int(sample.label.sum()):,}，"
        f"正例率 {sample.label.mean():.4%}", flush=True,
    )
    del engine, w2v, bpr
    gc.collect()
    return sample


def forward_oof(samples: pd.DataFrame, seed: int, decay: float):
    months = sorted(samples.snapshot_month.unique())
    validation_months = months[2:]
    folds = []
    config_fold_metrics = {name: [] for name in LGBM_CONFIGS}
    cat_metrics, lr_metrics = [], []
    cat_columns = ["user_id", "item_id", *COMPETITION_FEATURES]

    for fold_index, validation_month in enumerate(validation_months):
        train = samples.loc[samples.snapshot_month < validation_month].copy()
        valid = samples.loc[samples.snapshot_month == validation_month].copy()
        fold = valid[["user_id", "item_id", "label", "snapshot_month"]].copy()
        print(f"v9 OOF {validation_month}: train={len(train):,}, valid={len(valid):,}", flush=True)
        for name, (parameters, positive_weight) in LGBM_CONFIGS.items():
            model = make_lgbm(seed + fold_index, parameters)
            weights = temporal_weights(train, train.snapshot_month.max(), decay, positive_weight)
            model.fit(train[COMPETITION_FEATURES], train.label, sample_weight=weights)
            fold[f"lgb__{name}"] = percentile(model.predict_proba(valid[COMPETITION_FEATURES])[:, 1])
            config_fold_metrics[name].append(metric_bundle(valid, fold[f"lgb__{name}"].to_numpy()))

        lr = make_lr(seed + fold_index)
        lr_weights = temporal_weights(train, train.snapshot_month.max(), decay, 2.0)
        lr.fit(train[COMPETITION_FEATURES], train.label, logisticregression__sample_weight=lr_weights)
        fold["lr"] = percentile(lr.predict_proba(valid[COMPETITION_FEATURES])[:, 1])
        lr_metrics.append(metric_bundle(valid, fold.lr.to_numpy()))

        cat = make_catboost(seed + fold_index)
        cat_weights = temporal_weights(train, train.snapshot_month.max(), decay, 2.0)
        cat.fit(
            train[cat_columns], train.label,
            cat_features=["user_id", "item_id"], sample_weight=cat_weights,
        )
        fold["cat"] = percentile(cat.predict_proba(valid[cat_columns])[:, 1])
        cat_metrics.append(metric_bundle(valid, fold.cat.to_numpy()))
        for component, feature in DIRECT_COMPONENTS.items():
            fold[component] = percentile(valid[feature].to_numpy(float))
        folds.append(fold)

    mean_metrics = {
        name: {key: float(np.mean([fold[key] for fold in values])) for key in values[0]}
        for name, values in config_fold_metrics.items()
    }
    best_config = max(mean_metrics, key=lambda name: mean_metrics[name]["robust_score"])
    oof = pd.concat(folds, ignore_index=True)
    oof["lgb"] = oof[f"lgb__{best_config}"]
    component_metrics = {
        "lgb": metric_bundle(oof, oof.lgb.to_numpy()),
        "lr": metric_bundle(oof, oof.lr.to_numpy()),
        "cat": metric_bundle(oof, oof.cat.to_numpy()),
        **{name: metric_bundle(oof, oof[name].to_numpy()) for name in DIRECT_COMPONENTS},
    }

    components = ["lgb", "cat", "lr", *DIRECT_COMPONENTS]
    matrix = oof[components].to_numpy(float)
    rng = np.random.default_rng(seed + 909)
    weight_candidates = [np.eye(len(components))[i] for i in range(len(components))]
    weight_candidates.extend([
        np.asarray([0.75, 0.10, 0.15, 0, 0, 0, 0], dtype=float),
        np.asarray([0.65, 0.20, 0.15, 0, 0, 0, 0], dtype=float),
        np.asarray([0.80, 0.00, 0.15, 0.05, 0, 0, 0], dtype=float),
    ])
    weight_candidates.extend(rng.dirichlet([10, 2.5, 2.5, 0.8, 0.8, 0.5, 0.8], size=260))
    screened = sorted(
        ((fast_auc(oof.label.to_numpy(), matrix @ weights), weights) for weights in weight_candidates),
        key=lambda value: value[0], reverse=True,
    )
    finalists = screened[:55] + [(fast_auc(oof.label.to_numpy(), matrix @ weights), weights) for weights in weight_candidates[:len(components)]]
    best = None
    for _, weights in finalists:
        scores = matrix @ weights
        metrics = metric_bundle(oof, scores)
        if best is None or metrics["robust_score"] > best[0]["robust_score"]:
            best = (metrics, weights.copy(), scores)
    blend_metrics, blend_weights, blend_scores = best
    oof["blend"] = blend_scores
    blend_spec = {
        "rank_scope": "global",
        "components": {
            name: float(weight) for name, weight in zip(components, blend_weights) if weight >= 1e-6
        },
        "selection_objective": "0.55*AUC + 0.30*weighted_gAUC + 0.15*macro_gAUC",
        "oof_metrics": blend_metrics,
    }
    return oof, mean_metrics, component_metrics, best_config, blend_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练候选集对齐历史回放 v9")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-models-dir", type=Path, default=DEFAULT_SOURCE_MODELS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--model-version", default="traincsv-v9")
    parser.add_argument("--first-label-month", default="2010-06")
    parser.add_argument("--last-label-month", default="2010-10")
    parser.add_argument("--temporal-decay", type=float, default=0.82)
    parser.add_argument("--bpr-epochs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    clean, catalog = load_training_data(args.train_csv)
    candidates = pd.read_csv(args.candidate_csv, dtype=str, keep_default_na=False)
    if list(candidates.columns) != ["user_id", "item_id"]:
        raise ValueError("候选集必须严格为 user_id,item_id 两列")
    item_weights = candidates.item_id.value_counts().astype(float).to_dict()
    order_times = clean.groupby(["user_id", "invoice_no"]).transaction_time.transform("max")
    months = pd.period_range(args.first_label_month, args.last_label_month, freq="M")
    cfg = replace(
        Config(), history_end=f"{args.first_label_month}-01",
        test_start=str((clean.transaction_time.max() + pd.Timedelta(days=1)).date()), eval_users=0,
    )
    snapshots = [
        build_snapshot(clean, catalog, order_times, candidates, item_weights, month, cfg, args.bpr_epochs)
        for month in months
    ]
    samples = pd.concat(snapshots, ignore_index=True)
    samples.to_csv(args.outputs_dir / "candidate_aligned_samples.csv", index=False)

    oof, config_metrics, component_metrics, best_config, blend_spec = forward_oof(
        samples, cfg.seed, args.temporal_decay
    )
    oof.to_csv(args.outputs_dir / "candidate_aligned_oof.csv", index=False)
    print("v9 LightGBM OOF：" + json.dumps(config_metrics, ensure_ascii=False), flush=True)
    print("v9 组件 OOF：" + json.dumps(component_metrics, ensure_ascii=False), flush=True)
    print("v9 融合：" + json.dumps(blend_spec, ensure_ascii=False), flush=True)

    parameters, positive_weight = LGBM_CONFIGS[best_config]
    final_weights = temporal_weights(samples, samples.snapshot_month.max(), args.temporal_decay, positive_weight)
    lgb = make_lgbm(cfg.seed, parameters)
    lgb.fit(samples[COMPETITION_FEATURES], samples.label, sample_weight=final_weights)
    lr = make_lr(cfg.seed)
    lr.fit(
        samples[COMPETITION_FEATURES], samples.label,
        logisticregression__sample_weight=temporal_weights(
            samples, samples.snapshot_month.max(), args.temporal_decay, 2.0
        ),
    )
    cat_columns = ["user_id", "item_id", *COMPETITION_FEATURES]
    cat = make_catboost(cfg.seed)
    cat.fit(
        samples[cat_columns], samples.label,
        cat_features=["user_id", "item_id"],
        sample_weight=temporal_weights(samples, samples.snapshot_month.max(), args.temporal_decay, 2.0),
    )
    rankers = {"lightgbm": lgb, "lr": lr, "catboost": cat}

    # Reuse the already trained full-history vector/BPR engine from v8.  Only
    # candidate-aligned statistics and v9 rankers are new, avoiding an unnecessary
    # second full BPR run.
    final_engine = BaselineEngine.load(args.source_models_dir)
    attach_competition_state(final_engine, clean, item_weights, candidate_frame=candidates)
    final_engine.save(args.models_dir)
    source_w2v = args.source_models_dir / "item2vec.model"
    if source_w2v.exists():
        shutil.copy2(source_w2v, args.models_dir / "item2vec.model")
    with (args.models_dir / "rankers.pkl").open("wb") as stream:
        pickle.dump(rankers, stream)

    metadata = {
        "model_family_version": args.model_version,
        "submission_versions": {
            "lightgbm": f"{args.model_version}-lgbm",
            "lr": f"{args.model_version}-lr",
            "catboost": f"{args.model_version}-catboost",
            "oof_blend": f"{args.model_version}-oofblend",
        },
        "training_sha256": file_sha256(args.train_csv),
        "candidate_sha256": file_sha256(args.candidate_csv),
        "candidate_used_as_labels": False,
        "candidate_alignment": "fixed candidate rows replayed; labels exclusively from train.csv monthly purchases",
        "label_months": [str(month) for month in months],
        "sample_rows": int(len(samples)),
        "positive_rows": int(samples.label.sum()),
        "positive_rate": float(samples.label.mean()),
        "features": COMPETITION_FEATURES,
        "selected_lightgbm_config": best_config,
        "lightgbm_oof_metrics": config_metrics,
        "component_oof_metrics": component_metrics,
        "blend_spec": blend_spec,
        "source_full_engine": str(args.source_models_dir.resolve()),
        "seconds": time.perf_counter() - started,
    }
    (args.models_dir / "model_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
