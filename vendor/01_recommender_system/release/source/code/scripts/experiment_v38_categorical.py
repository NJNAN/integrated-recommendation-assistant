"""Bounded time-forward native-categorical experiment using course train.csv only.

Uses v9 snapshots June--October, adds v15 history features, and validates August,
September, October on strictly earlier labeled snapshots. Category dictionaries
come from each training fold; month and row order are never predictive inputs.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.competition import COMPETITION_FEATURES
from baseline.competition_metrics_v16 import competition_metrics, monthly_competition_metrics
from baseline.engine import BaselineEngine
from baseline.features_v15 import INTERACTION_FEATURES_V15, append_v15_interactions
from baseline.seasonal_features import SEASONAL_FEATURES, append_seasonal_frame, build_seasonal_state
from scripts.make_submission_v15 import complete_feature_frame
from scripts.make_submission_v11_season import read_candidates, validate_submission
from scripts.train_competition import file_sha256, load_training_data
from scripts.train_competition_v9 import temporal_weights

OUT = ROOT / "outputs/competition_v38_categorical"
MODELS = ROOT / "models/competition_v38_categorical"
KEYS = ["snapshot_month", "user_id", "item_id", "label"]
BASE_FEATURES = [*COMPETITION_FEATURES, *SEASONAL_FEATURES, *INTERACTION_FEATURES_V15]
CONFIGS = {
    "item_only": ["item_id"],
    "user_item": ["user_id", "item_id"],
}
PARAMS = dict(
    objective="binary", n_estimators=600, learning_rate=0.025,
    num_leaves=31, min_child_samples=100, reg_alpha=0.4, reg_lambda=8,
    max_cat_to_onehot=4, max_cat_threshold=32, cat_smooth=80, cat_l2=15,
    min_data_per_group=100, colsample_bytree=0.9, subsample=0.9,
    subsample_freq=1, max_bin=255, random_state=3800, n_jobs=3,
    deterministic=True, force_col_wise=True, verbosity=-1,
)


def encode(frame, categories, vocab=None):
    matrix = frame[BASE_FEATURES].astype(np.float32).copy()
    vocab = vocab or {col: sorted(frame[col].astype(str).unique()) for col in categories}
    for col in categories:
        matrix[col] = pd.Categorical(frame[col].astype(str), categories=vocab[col])
    return matrix, vocab


def fit(train, categories):
    matrix, vocab = encode(train, categories)
    model = LGBMClassifier(**PARAMS)
    weights = temporal_weights(train, train.snapshot_month.max(), 0.82, positive_weight=2.0)
    model.fit(matrix, train.label.astype(int), sample_weight=weights, categorical_feature=categories)
    return {"model": model, "vocab": vocab, "categories": categories,
            "features": BASE_FEATURES + categories}


def predict(package, frame):
    matrix, _ = encode(frame, package["categories"], package["vocab"])
    return package["model"].predict_proba(matrix)[:, 1]


def metric_record(frame, column):
    total = competition_metrics(frame, column)
    months = monthly_competition_metrics(frame, column)
    return {"pooled": total, "monthly": months,
            "mean_monthly_score": float(np.mean([v["score"] for v in months.values()])),
            "worst_month_score": float(min(v["score"] for v in months.values()))}


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    sample_path = ROOT / "outputs/competition_v9/candidate_aligned_samples.csv"
    samples = pd.read_csv(sample_path, dtype={"user_id": str, "item_id": str, "snapshot_month": str})
    assert sorted(samples.snapshot_month.unique()) == [f"2010-{m:02}" for m in range(6, 11)]
    assert not samples.duplicated(KEYS).any()
    assert not any(col in BASE_FEATURES for col in ["label", "snapshot_month", "_row", "index"])
    clean, _ = load_training_data(ROOT / "data/train.csv")
    assert clean.transaction_time.max() < pd.Timestamp("2010-11-01")
    enriched = []
    for month, part in samples.groupby("snapshot_month", sort=True):
        # The descriptor is also cutoff-safe; seasonal counts end March 1.
        history = clean.loc[clean.transaction_time < pd.Timestamp(month)]
        state = build_seasonal_state(history)
        enriched.append(append_v15_interactions(append_seasonal_frame(part, state)))
    samples = pd.concat(enriched, ignore_index=True)
    assert set(BASE_FEATURES).issubset(samples.columns)
    oof = pd.read_csv(ROOT / "outputs/competition_v15/candidate_aligned_oof_v15.csv",
                      dtype={"user_id": str, "item_id": str, "snapshot_month": str},
                      usecols=KEYS + ["blend_v15"])
    all_metrics = {"blend_v15": metric_record(oof, "blend_v15")}
    print(f"Rows={len(samples):,}; features={len(BASE_FEATURES)}; baseline="
          f"{all_metrics['blend_v15']['mean_monthly_score']:.6f}", flush=True)
    fold_audit = []
    for name, categories in CONFIGS.items():
        folds = []
        for month in ["2010-08", "2010-09", "2010-10"]:
            t0 = time.perf_counter()
            train = samples.loc[samples.snapshot_month < month]
            valid = samples.loc[samples.snapshot_month == month]
            assert train.snapshot_month.max() < valid.snapshot_month.min()
            package = fit(train, categories)
            fold = valid[KEYS].copy()
            fold[name] = predict(package, valid)
            folds.append(fold)
            fold_audit.append({"config": name, "valid_month": month,
                               "train_months": sorted(train.snapshot_month.unique()),
                               "training_rows": len(train), "validation_rows": len(valid),
                               "category_cardinality": {k: len(v) for k, v in package["vocab"].items()},
                               "seconds": time.perf_counter() - t0})
            print(json.dumps(fold_audit[-1]), flush=True)
        scored = pd.concat(folds, ignore_index=True)
        oof = oof.merge(scored, on=KEYS, how="left", validate="one_to_one")
        assert oof[name].notna().all()
        all_metrics[name] = metric_record(oof, name)
        for mode, group in [("user", ["snapshot_month", "user_id"]), ("month", ["snapshot_month"])]:
            base = oof.groupby(group, sort=False).blend_v15.rank(pct=True)
            extra = oof.groupby(group, sort=False)[name].rank(pct=True)
            for weight in [0.1, 0.2, 0.3]:
                col = f"{name}_{mode}_w{int(weight*100):02}"
                oof[col] = (1 - weight) * base + weight * extra
                all_metrics[col] = metric_record(oof, col)
                delta = all_metrics[col]["mean_monthly_score"] - all_metrics["blend_v15"]["mean_monthly_score"]
                print(f"{col}: score={all_metrics[col]['mean_monthly_score']:.6f} delta={delta:+.6f}", flush=True)
        oof.to_csv(OUT / "candidate_aligned_oof_v38.csv", index=False)
        (OUT / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    baseline = all_metrics["blend_v15"]
    summary = []
    for name, record in all_metrics.items():
        deltas = {month: m["score"] - baseline["monthly"][month]["score"]
                  for month, m in record["monthly"].items()}
        summary.append({"component": name, "mean_monthly_score": record["mean_monthly_score"],
                        "pooled_score": record["pooled"]["score"],
                        "mean_delta": record["mean_monthly_score"] - baseline["mean_monthly_score"],
                        **{f"delta_{m}": d for m, d in deltas.items()}})
    pd.DataFrame(summary).sort_values("mean_delta", ascending=False).to_csv(OUT / "summary.csv", index=False)

    # At most two final fits; retain both for bounded, transparent online testing.
    candidates = read_candidates(ROOT / "data/sample_candidates.csv")
    print("Building production candidate features", flush=True)
    engine = BaselineEngine.load(ROOT / "models/competition_v11_season")
    full, known = complete_feature_frame(candidates, engine, build_seasonal_state(clean))
    full.to_pickle(OUT / "production_v15_features.pkl")
    np.save(OUT / "production_known_mask.npy", known)
    print(f"Production features cached: {OUT / 'production_v15_features.pkl'}", flush=True)
    v15_live = pd.read_csv(ROOT / "outputs/submissions/submission_traincsv_v15_activity_moe_blend.csv",
                           dtype={"user_id": str, "item_id": str})
    live_base = candidates.merge(v15_live, on=["user_id", "item_id"], how="left", validate="one_to_one")
    assert live_base.score.notna().all()
    final_columns = candidates.copy()
    final_columns["v15"] = live_base.score.to_numpy(float)
    for name, categories in CONFIGS.items():
        package = fit(samples, categories)
        package["params"] = PARAMS
        with (MODELS / f"{name}.pkl").open("wb") as stream:
            pickle.dump(package, stream)
        scores = np.zeros(len(candidates), dtype=float)
        scores[known] = predict(package, full)
        final_columns[name] = scores
        pure = candidates.copy()
        pure["score"] = scores
        validate_submission(candidates, pure)
        pure.to_csv(OUT / f"submission_v38_{name}.csv", index=False)
        for mode in ["user", "month"]:
            if mode == "user":
                base = final_columns.groupby("user_id", sort=False).v15.rank(pct=True)
                extra = final_columns.groupby("user_id", sort=False)[name].rank(pct=True)
            else:
                base = final_columns.v15.rank(pct=True)
                extra = final_columns[name].rank(pct=True)
            for weight in [0.1, 0.2, 0.3]:
                submission = candidates.copy()
                submission["score"] = (1 - weight) * base + weight * extra
                submission.loc[~known, "score"] = 0
                validate_submission(candidates, submission)
                submission.to_csv(OUT / f"submission_v38_{name}_{mode}_w{int(weight*100):02}.csv", index=False)
    final_columns.to_csv(OUT / "production_components.csv", index=False)
    meta = {"params": PARAMS, "configs": CONFIGS, "features": BASE_FEATURES,
            "folds": fold_audit, "data_source": "course code/data/train.csv only",
            "train_sha256": file_sha256(ROOT / "data/train.csv"),
            "samples_sha256": file_sha256(sample_path),
            "candidate_sha256": file_sha256(ROOT / "data/sample_candidates.csv"),
            "leakage_controls": ["Only June--October labels; no November labels",
                                 "Strict forward outer August, September, October",
                                 "Seasonal history and descriptions cut off at each snapshot",
                                 "Only explicit feature allowlist; no row or index features",
                                 "Training-fold categorical vocabularies only"],
            "metric_note": "Local project evaluator: 0.4 GAUC + 0.4 NDCG@10 + 0.2 Recall@10; not official server code",
            "selection_note": "Two prespecified models; all 0.1/0.2/0.3 blends retained. OOF blend comparison is not nested selection.",
            "production_rows": len(candidates), "unknown_item_rows": int((~known).sum()),
            "seconds": time.perf_counter() - started}
    (OUT / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).sort_values("mean_delta", ascending=False).to_string(index=False), flush=True)
    print(f"Complete in {meta['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
