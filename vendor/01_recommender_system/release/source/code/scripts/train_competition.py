"""使用老师提供的 data/train.csv 训练比赛专用模型。

训练设计：
1. 2009-12 至 2010-08 构造历史特征；2010-09 至 2010-10 作为未来购买标签；
2. 对未来真实购买对取正例，并为每位用户从历史商品全集抽取 1:5 负例；
3. 训练 LR 与 LightGBM 精排器；
4. 用全部 train.csv 重建最终 Item2Vec、ItemCF、用户向量及统计特征；
5. 产物独立写入 models/competition，不覆盖课程基线模型。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.config import Config  # noqa: E402
from baseline.data import clean_data, sequences  # noqa: E402
from baseline.engine import BaselineEngine, FEATURES, train_item2vec  # noqa: E402
from baseline.competition import (  # noqa: E402
    COMPETITION_FEATURES,
    attach_competition_state,
    competition_features,
)

CATBOOST_FEATURES = ["user_id", "item_id", *COMPETITION_FEATURES]


DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "sample_candidates.csv"
DEFAULT_MODELS = ROOT / "models" / "competition"
DEFAULT_OUTPUTS = ROOT / "outputs" / "competition"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        path,
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
    )
    required = {
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"train.csv 缺少字段：{sorted(missing)}")

    # train.csv 自带英文商品描述，但没有课程主数据中的中文名和类目。
    products = raw[["StockCode", "Description"]].copy()
    products["Chinese_Description"] = ""
    products["Product_Category"] = "未知"
    clean, catalog, _, _ = clean_data(raw, products)
    if clean.empty:
        raise ValueError("train.csv 清洗后为空")
    return clean, catalog


def temporal_parts(clean: pd.DataFrame, rank_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundary = pd.Timestamp(rank_start)
    order_times = clean.groupby(["user_id", "invoice_no"]).transaction_time.transform("max")
    history = clean.loc[order_times < boundary].copy()
    labels = clean.loc[order_times >= boundary].copy()
    if history.empty or labels.empty:
        raise ValueError("时间切分出现空窗口，请调整 --rank-start")
    return history, labels


def build_rank_samples(
    engine: BaselineEngine,
    future: pd.DataFrame,
    negative_ratio: int,
    seed: int,
    item_weights: dict[str, float] | None = None,
    prior_negative_fraction: float = 0.0,
) -> pd.DataFrame:
    """构造任意 user-item 对训练集，正例不要求先被内部召回。"""
    rng = np.random.default_rng(seed)
    universe = np.asarray(engine.ids)
    rows: list[pd.DataFrame] = []

    for number, (uid, group) in enumerate(future.groupby("user_id", sort=True)):
        uid = str(uid)
        if uid not in engine.histories:
            continue
        positives = sorted(set(group.item_id.astype(str)) & set(engine.pos))
        if not positives:
            continue
        positive_set = set(positives)
        eligible = universe[~np.isin(universe, positives)]
        negative_count = min(len(eligible), len(positives) * negative_ratio)

        # 老师候选集中约 15% 是用户买过的商品。显式匹配这个比例，避免训练负例
        # 几乎全是陌生商品、而考试候选包含大量“买过但这次不会买”的简单分布偏差。
        prior_pool = np.asarray(sorted(set(engine.histories.get(uid, [])) - positive_set))
        prior_count = min(len(prior_pool), int(round(negative_count * prior_negative_fraction)))

        def choose(pool: np.ndarray, count: int) -> list[str]:
            if count <= 0 or len(pool) == 0:
                return []
            probabilities = None
            if item_weights:
                raw_weights = np.asarray([item_weights.get(str(item), 0.1) for item in pool], dtype=float)
                probabilities = raw_weights / raw_weights.sum()
            return rng.choice(pool, min(count, len(pool)), replace=False, p=probabilities).tolist()

        prior_negatives = choose(prior_pool, prior_count)
        remaining_pool = eligible[~np.isin(eligible, prior_negatives)]
        negatives = prior_negatives + choose(remaining_pool, negative_count - len(prior_negatives))
        item_ids = positives + negatives
        labels = np.r_[np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)]

        recalled, scores, _ = engine.candidates(uid)
        source_map = {row["item_id"]: row["sources"] for row in recalled}
        candidates = [{"item_id": item, "sources": source_map.get(item, [])} for item in item_ids]
        features = competition_features(engine, uid, candidates, scores)
        features.insert(0, "item_id", item_ids)
        features.insert(0, "user_id", uid)
        features["label"] = labels
        rows.append(features)
        if number and number % 500 == 0:
            print(f"已构造用户样本：{number}", flush=True)

    if not rows:
        raise ValueError("未构造出可训练的正负样本")
    result = pd.concat(rows, ignore_index=True)
    if result.label.nunique() != 2:
        raise ValueError("训练样本必须同时包含正例和负例")
    return result


def train_rankers(sample: pd.DataFrame, seed: int):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    lr.fit(sample[COMPETITION_FEATURES], sample.label)

    from lightgbm import LGBMClassifier

    lightgbm = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.04,
        min_child_samples=40,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    lightgbm.fit(sample[COMPETITION_FEATURES], sample.label)
    from lightgbm import LGBMRanker

    ranked_sample = sample.sort_values(["user_id", "item_id"], kind="stable").reset_index(drop=True)
    group = ranked_sample.groupby("user_id", sort=False).size().to_numpy()
    lgbm_ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=300,
        num_leaves=31, learning_rate=0.03, min_child_samples=30,
        reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
        lambdarank_truncation_level=100,
    )
    lgbm_ranker.fit(
        ranked_sample[COMPETITION_FEATURES], ranked_sample.label,
        group=group,
    )
    from catboost import CatBoostClassifier

    catboost = CatBoostClassifier(
        iterations=600, depth=8, learning_rate=0.04,
        loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=5.0,
        random_seed=seed, thread_count=4, verbose=False,
        allow_writing_files=False,
    )
    catboost.fit(
        sample[CATBOOST_FEATURES], sample.label,
        cat_features=["user_id", "item_id"],
    )
    return {
        "lr": lr, "lightgbm": lightgbm,
        "lgbm_ranker": lgbm_ranker, "catboost": catboost,
    }


def model_scores(model, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    return np.asarray(model.predict(frame), dtype=float)


def group_auc(frame: pd.DataFrame, scores: np.ndarray, weighted: bool = True) -> float:
    """按用户计算 AUC；默认按该用户候选行数加权。"""
    work = frame[["user_id", "label"]].copy()
    work["score"] = scores
    aucs, weights = [], []
    for _, group in work.groupby("user_id", sort=False):
        if group.label.nunique() < 2:
            continue
        aucs.append(roc_auc_score(group.label, group.score))
        weights.append(len(group) if weighted else 1)
    if not aucs:
        raise ValueError("没有同时包含正负标签的用户，无法计算 gAUC")
    return float(np.average(aucs, weights=weights))


def make_lightgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.04,
        min_child_samples=40, reg_lambda=1.0, random_state=seed,
        n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
    )


def compare_on_october(clean, catalog, cfg, negative_ratio: int, item_weights, prior_fraction) -> dict:
    """用 10 月作完全未参与拟合的时间验证，对比原特征与新增交叉特征。"""
    september = pd.Timestamp("2010-09-01")
    october = pd.Timestamp("2010-10-01")
    order_times = clean.groupby(["user_id", "invoice_no"]).transaction_time.transform("max")
    fit_history = clean.loc[order_times < september].copy()
    fit_labels = clean.loc[(order_times >= september) & (order_times < october)].copy()
    validation_history = clean.loc[order_times < october].copy()
    validation_labels = clean.loc[order_times >= october].copy()

    print("时间验证：训练 9 月单月标签，严格验证 10 月", flush=True)
    fit_w2v = train_item2vec(sequences(fit_history), cfg)
    fit_engine = BaselineEngine(fit_history, catalog, cfg, fit_w2v)
    attach_competition_state(fit_engine, fit_history, item_weights)
    fit_sample = build_rank_samples(fit_engine, fit_labels, negative_ratio, cfg.seed + 101)
    aware_fit_sample = build_rank_samples(
        fit_engine, fit_labels, negative_ratio, cfg.seed + 101,
        item_weights=item_weights, prior_negative_fraction=prior_fraction,
    )

    validation_w2v = train_item2vec(sequences(validation_history), cfg)
    validation_engine = BaselineEngine(validation_history, catalog, cfg, validation_w2v)
    attach_competition_state(validation_engine, validation_history, item_weights)
    validation_sample = build_rank_samples(
        validation_engine, validation_labels, negative_ratio, cfg.seed + 202,
        item_weights=item_weights, prior_negative_fraction=prior_fraction,
    )

    baseline = make_lightgbm(cfg.seed)
    enhanced = make_lightgbm(cfg.seed)
    candidate_aware = make_lightgbm(cfg.seed)
    baseline.fit(fit_sample[FEATURES], fit_sample.label)
    enhanced.fit(fit_sample[COMPETITION_FEATURES], fit_sample.label)
    candidate_aware.fit(aware_fit_sample[COMPETITION_FEATURES], aware_fit_sample.label)
    from lightgbm import LGBMRanker
    rank_fit = aware_fit_sample.sort_values(["user_id", "item_id"], kind="stable").reset_index(drop=True)
    rank_group = rank_fit.groupby("user_id", sort=False).size().to_numpy()
    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=300,
        num_leaves=31, learning_rate=0.03, min_child_samples=30,
        reg_lambda=1.0, random_state=cfg.seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True, lambdarank_truncation_level=100,
    )
    ranker.fit(rank_fit[COMPETITION_FEATURES], rank_fit.label, group=rank_group)
    from catboost import CatBoostClassifier

    catboost = CatBoostClassifier(
        iterations=1000, depth=8, learning_rate=0.04,
        loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=5.0,
        random_seed=cfg.seed, thread_count=4, verbose=False,
        allow_writing_files=False, od_type="Iter", od_wait=60,
    )
    catboost.fit(
        aware_fit_sample[CATBOOST_FEATURES], aware_fit_sample.label,
        cat_features=["user_id", "item_id"],
        eval_set=(validation_sample[CATBOOST_FEATURES], validation_sample.label),
        use_best_model=True,
    )
    candidate_scores = candidate_aware.predict_proba(validation_sample[COMPETITION_FEATURES])[:, 1]
    ranker_scores = ranker.predict(validation_sample[COMPETITION_FEATURES])
    catboost_scores = catboost.predict_proba(validation_sample[CATBOOST_FEATURES])[:, 1]
    return {
        "validation_month": "2010-10",
        "fit_rows": int(len(fit_sample)),
        "validation_rows": int(len(validation_sample)),
        "baseline_auc": float(roc_auc_score(
            validation_sample.label, baseline.predict_proba(validation_sample[FEATURES])[:, 1]
        )),
        "enhanced_auc": float(roc_auc_score(
            validation_sample.label,
            enhanced.predict_proba(validation_sample[COMPETITION_FEATURES])[:, 1],
        )),
        "candidate_aware_auc": float(roc_auc_score(
            validation_sample.label, candidate_scores,
        )),
        "candidate_aware_gauc": group_auc(validation_sample, candidate_scores),
        "lgbm_ranker_auc": float(roc_auc_score(validation_sample.label, ranker_scores)),
        "lgbm_ranker_gauc": group_auc(validation_sample, ranker_scores),
        "catboost_auc": float(roc_auc_score(validation_sample.label, catboost_scores)),
        "catboost_gauc": group_auc(validation_sample, catboost_scores),
        "catboost_best_iteration": int(catboost.get_best_iteration()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 data/train.csv 训练比赛模型")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rank-start", default="2010-10-01", help="监督标签窗口起点")
    parser.add_argument("--model-version", default="traincsv-v1")
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--skip-validation", action="store_true", help="跳过 10 月时间验证")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.negative_ratio < 1:
        raise ValueError("--negative-ratio 必须至少为 1")
    started = time.perf_counter()
    DEFAULT_MODELS.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUTS.mkdir(parents=True, exist_ok=True)

    clean, catalog = load_training_data(args.train_csv)
    candidate_frame = pd.read_csv(args.candidate_csv, dtype=str, keep_default_na=False)
    if list(candidate_frame.columns) != ["user_id", "item_id"]:
        raise ValueError("候选集必须严格为 user_id,item_id 两列")
    item_weights = candidate_frame.item_id.value_counts().astype(float).to_dict()
    purchased_pairs = set(zip(clean.user_id.astype(str), clean.item_id.astype(str)))
    prior_fraction = float(np.mean([
        (str(uid), str(item)) in purchased_pairs
        for uid, item in zip(candidate_frame.user_id, candidate_frame.item_id)
    ]))
    history, future = temporal_parts(clean, args.rank_start)
    cfg = replace(
        Config(),
        history_end=args.rank_start,
        test_start=str((clean.transaction_time.max() + pd.Timedelta(days=1)).date()),
        eval_users=0,
    )

    validation = None if args.skip_validation else compare_on_october(
        clean, catalog, cfg, args.negative_ratio, item_weights, prior_fraction
    )
    if validation:
        validation["feature_auc_gain"] = validation["enhanced_auc"] - validation["baseline_auc"]
        validation["candidate_aware_auc_gain"] = validation["candidate_aware_auc"] - validation["enhanced_auc"]
        print("时间验证结果：" + json.dumps(validation, ensure_ascii=False), flush=True)

    print(f"训练精排特征引擎：历史 {len(history):,} 行，标签 {len(future):,} 行", flush=True)
    history_w2v = train_item2vec(sequences(history), cfg)
    history_engine = BaselineEngine(history, catalog, cfg, history_w2v)
    attach_competition_state(history_engine, history, item_weights)
    sample = build_rank_samples(
        history_engine, future, args.negative_ratio, cfg.seed,
        item_weights=item_weights, prior_negative_fraction=prior_fraction,
    )
    rankers = train_rankers(sample, cfg.seed)

    print(f"使用全部 train.csv 重建最终推荐引擎：{len(clean):,} 行", flush=True)
    final_w2v = train_item2vec(sequences(clean), cfg)
    final_engine = BaselineEngine(clean, catalog, cfg, final_w2v)
    attach_competition_state(final_engine, clean, item_weights)
    final_engine.save(DEFAULT_MODELS)
    final_w2v.save(str(DEFAULT_MODELS / "item2vec.model"))
    with (DEFAULT_MODELS / "rankers.pkl").open("wb") as stream:
        pickle.dump(rankers, stream)

    sample.to_csv(DEFAULT_OUTPUTS / "rank_train_samples.csv", index=False)
    auc = {}
    for name, model in rankers.items():
        columns = CATBOOST_FEATURES if name == "catboost" else COMPETITION_FEATURES
        auc[name] = float(roc_auc_score(sample.label, model_scores(model, sample[columns])))
    metadata = {
        "model_family_version": args.model_version,
        "submission_versions": {
            "lr": f"{args.model_version}-lr",
            "lightgbm": f"{args.model_version}-lgbm",
            "lgbm_ranker": f"{args.model_version}-ranker",
            "catboost": f"{args.model_version}-catboost",
        },
        "training_file": str(args.train_csv.resolve()),
        "training_sha256": file_sha256(args.train_csv),
        "candidate_file": str(args.candidate_csv.resolve()),
        "candidate_sha256": file_sha256(args.candidate_csv),
        "candidate_prior_bought_fraction": prior_fraction,
        "raw_rows": int(len(pd.read_csv(args.train_csv, usecols=["CustomerID"]))),
        "clean_rows": int(len(clean)),
        "date_min": str(clean.transaction_time.min()),
        "date_max": str(clean.transaction_time.max()),
        "rank_label_start": args.rank_start,
        "rank_history_rows": int(len(history)),
        "rank_label_rows": int(len(future)),
        "rank_sample_rows": int(len(sample)),
        "positive_rows": int(sample.label.sum()),
        "negative_rows": int((sample.label == 0).sum()),
        "users": int(clean.user_id.nunique()),
        "items": int(clean.item_id.nunique()),
        "features": COMPETITION_FEATURES,
        "training_auc_self_check": auc,
        "temporal_validation": validation,
        "seconds": time.perf_counter() - started,
    }
    (DEFAULT_MODELS / "model_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 持久化回读校验，防止模型文件与向量索引错配。
    restored = BaselineEngine.load(DEFAULT_MODELS)
    if len(restored.ids) != clean.item_id.nunique():
        raise ValueError("最终模型商品数与训练数据不一致")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
