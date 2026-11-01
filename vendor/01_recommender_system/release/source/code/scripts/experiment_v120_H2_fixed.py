# -*- coding: utf-8 -*-
"""v120：修复两个已确认缺陷后，重建 H2 主链的 4 个特征 pickle。

修复的两个缺陷
--------------
缺陷 1（引擎时间穿越）
    v95 对五个历史窗口加载同一个冻结引擎 `models/competition_v11_season`，
    而该引擎是用截至 2010-10-31 的全部数据建成的。于是 histories / orders /
    popularity / vector_neighbors / cf_neighbors / user_vectors /
    competition_time_decay_itemcf / competition_bpr_* 这些**未被
    attach_competition_state 覆盖**的字段，在预测 6—10 月标签时读到了未来月份。
    修复：每个窗口用该 cutoff 之前的 hist 重建一个完整引擎，配方与
    train_competition_v8.py 的最终引擎一致：
        train_item2vec(sequences(hist)) -> BaselineEngine(hist, catalog, cfg, w2v)
        -> attach_competition_state(eng, hist, item_weights, candidate_frame)
        -> train_time_decay_bpr(hist, ...) -> attach_bpr_state

    11 月（打分目标月）不需要重建：冻结引擎恰好等于"截至 2010-10-31 的全部数据"，
    对 11 月而言是正确的。

缺陷 2（行错位）
    make_submission_v15.complete_feature_frame 会丢掉 item 不在引擎池里的行，
    下游却按 DataFrame 行号 pd.concat(axis=1) 拼接，导致 290,581 行错位。
    修复：保留全部候选行，未知商品的特征填零；随后与标签按
    (window, user_id, item_id) 键合并，并断言行数、键顺序、唯一性。

输出（与 v95 同名同格式，下游 v102/v104/v105/v106 可直接切换 IN 目录）
    our_feats_train.pkl   (291,025, 73)   行序 = 候选顺序 × 5 窗口
    signals_train.pkl     (291,025, 13)
    our_feats_nov.pkl     ( 58,205, 73)
    signals_nov.pkl       ( 58,205, 13)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "0.7912的模型/recommendation"
CACHE = ROOT / "outputs/competition_v65_capacity/cache"
OUT = ROOT / "outputs/competition_v120_H2_fixed"
TOP_BOUGHT = 50
BPR_EPOCHS = 6

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

from baseline.bpr import attach_bpr_state, train_time_decay_bpr          # noqa: E402
from baseline.competition import (                                        # noqa: E402
    COMPETITION_FEATURES, attach_competition_state, competition_features,
)
from baseline.config import Config                                        # noqa: E402
from baseline.data import sequences                                       # noqa: E402
from baseline.engine import BaselineEngine, train_item2vec                # noqa: E402
from baseline.features_v15 import (                                      # noqa: E402
    INTERACTION_FEATURES_V15, append_v15_interactions,
)
from baseline.seasonal_features import (                                  # noqa: E402
    SEASONAL_FEATURES, add_seasonal_columns, build_seasonal_state,
)
from scripts.signals_third_family import SIGNAL_COLS, build_signals       # noqa: E402
from scripts.train_competition import load_training_data                  # noqa: E402

OUR_FEATS = [*COMPETITION_FEATURES, *SEASONAL_FEATURES, *INTERACTION_FEATURES_V15]
WINDOWS = [("2010-06", "2010-07-01 16:26:00"), ("2010-07", "2010-08-01 16:26:00"),
           ("2010-08", "2010-09-01 16:26:00"), ("2010-09", "2010-10-01 16:26:00"),
           ("2010-10", "2010-10-31 16:26:00")]
FROZEN_ENGINE = ROOT / "models/competition_v11_season"


def build_window_engine(hist: pd.DataFrame, catalog: pd.DataFrame,
                        item_weights: dict, cutoff: pd.Timestamp, tag: str,
                        candidates: pd.DataFrame):
    """用 hist 从零重建一个与该窗口一致的引擎（复刻 v8 最终引擎配方）。

    注意 candidates 必须传入：attach_competition_state 只有拿到
    candidate_frame 才会计算 7 个候选统计特征
    （candidate_user/item/pair 的历史正例率、活跃月率、lift、组大小、先验占比），
    否则它们会全部恒为 0 —— 旧 v95 是传了的，这里是修复后的必需参数。
    """
    cfg = replace(Config(), history_end=str(cutoff.date()),
                  test_start="2010-11-01", eval_users=0)
    w2v = train_item2vec(sequences(hist), cfg)
    eng = BaselineEngine(hist, catalog, cfg, w2v)
    attach_competition_state(eng, hist, item_weights, candidate_frame=candidates)
    bpr = train_time_decay_bpr(hist, item_weights, seed=cfg.seed, epochs=BPR_EPOCHS)
    attach_bpr_state(eng, bpr)
    cand_nonzero = sum(1 for v in eng.competition_candidate_item_positive_rate.values() if v)
    if cand_nonzero == 0:
        raise AssertionError(f"[{tag}] 候选统计特征全为空 —— candidate_frame 未生效")
    print(f"    [{tag}] 引擎重建完成：商品 {len(eng.ids)} | 用户 {len(eng.users)} "
          f"| 候选统计非零 {cand_nonzero} | BPR loss {bpr.final_loss:.4f}", flush=True)
    return eng


def full_feature_frame(candidates: pd.DataFrame, engine, seasonal_state):
    """保留全部候选行；未知商品（不在 engine.pos）的特征整行填零。

    返回 (全长特征帧, 已知掩码)，行序严格等于 candidates 的原始行序。
    """
    known_all = candidates.item_id.isin(engine.pos).to_numpy()
    frames = []
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
        full.insert(0, "_row", known.index.to_numpy())
        frames.append(full)

    got = (pd.concat(frames, ignore_index=True)
           .sort_values("_row").reset_index(drop=True)) if frames else pd.DataFrame()

    out = pd.DataFrame(0.0, index=np.arange(len(candidates)), columns=OUR_FEATS)
    if len(got):
        out.loc[got["_row"].to_numpy(int), :] = got[OUR_FEATS].astype(float).to_numpy()
    return out, known_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-users", type=int, default=0,
                    help=">0 时只跑前 N 个用户做冒烟测试")
    ap.add_argument("--skip-engine-rebuild", action="store_true",
                    help="仅用于验证对齐逻辑（会退回旧行为，产物不可用）")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    clean, catalog = load_training_data(ROOT / "data/train.csv")
    assert clean.transaction_time.max() < pd.Timestamp("2010-11-01")
    cands = pd.read_csv(ROOT / "data/sample_candidates.csv",
                        dtype={"user_id": str, "item_id": str})
    if args.smoke_users:
        keep = sorted(cands.user_id.unique())[: args.smoke_users]
        cands = cands[cands.user_id.isin(keep)].reset_index(drop=True)
        print(f"[冒烟] 只跑 {len(keep)} 个用户 / {len(cands)} 行", flush=True)
    item_weights = cands.item_id.value_counts().astype(float).to_dict()

    # ---------- 5 个历史窗口：逐窗口重建引擎 ----------
    our_blocks, sig_blocks, key_blocks, unknown_blocks = [], [], [], []
    for name, T in WINDOWS:
        cutoff = pd.Timestamp(T) - pd.Timedelta(days=30)
        hist = clean[clean.transaction_time < cutoff]
        assert hist.transaction_time.max() < cutoff
        print(f"  [{name}] cutoff={cutoff}  历史 {len(hist):,} 行", flush=True)

        if args.skip_engine_rebuild:
            eng = BaselineEngine.load(FROZEN_ENGINE)
            attach_competition_state(eng, hist, item_weights)
        else:
            eng = build_window_engine(hist, catalog, item_weights, cutoff, name, cands)

        feats, known = full_feature_frame(cands, eng, build_seasonal_state(hist))
        our_blocks.append(feats)
        sig_blocks.append(build_signals(cands.copy(), hist, top_bought=TOP_BOUGHT))
        key_blocks.append(pd.DataFrame({
            "_window": name,
            "user_id": cands.user_id.to_numpy(),
            "item_id": cands.item_id.to_numpy(),
        }))
        unknown_blocks.append(~known)
        print(f"  [{name}] 特征完成：该窗口未知商品 {int((~known).sum()):,} 行已填零 "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- 11 月：用冻结引擎（恰为“截至 2010-10-31 的全部数据”） ----------
    print("  [2010-11] 使用冻结引擎（对 11 月而言是正确口径）", flush=True)
    eng_nov = BaselineEngine.load(FROZEN_ENGINE)
    attach_competition_state(eng_nov, clean, item_weights)
    nov_feats, nov_known = full_feature_frame(cands, eng_nov, build_seasonal_state(clean))
    sig_nov = build_signals(cands.copy(), clean, top_bought=TOP_BOUGHT)
    print(f"  [2010-11] 特征完成：已知 {int(nov_known.sum()):,} / {len(nov_known):,} 行 "
          f"({time.time()-t0:.0f}s)", flush=True)

    OUR_TR = pd.concat(our_blocks, ignore_index=True)
    SIG_TR = pd.concat(sig_blocks, ignore_index=True)
    KEYS_TR = pd.concat(key_blocks, ignore_index=True)

    # ---------- 断言 1：行数 / 唯一性 / 与标签键完全一致 ----------
    tr = pd.read_pickle(CACHE / "train_win5.pkl")
    if args.smoke_users:
        print("\n  [冒烟模式] 跳过与全量标签的对齐断言", flush=True)
    else:
        assert len(OUR_TR) == len(tr) == 291025, f"行数不符 {len(OUR_TR)} vs {len(tr)}"
        assert not KEYS_TR.duplicated(["_window", "user_id", "item_id"]).any(), "特征键有重复"
        tr_k = tr[["_window", "user_id", "item_id"]].astype(str).reset_index(drop=True)
        our_k = KEYS_TR.astype(str).reset_index(drop=True)
        same = (tr_k.to_numpy() == our_k.to_numpy()).all()
        print(f"\n  [断言1] 特征键与标签键逐行一致：{same}", flush=True)
        if not same:
            diff = (tr_k.to_numpy() != our_k.to_numpy()).any(axis=1)
            print(f"          不一致行数 {int(diff.sum())}，首个不一致位置 {int(np.argmax(diff))}")
        assert same, "特征键与标签键不一致，禁止继续"

        assert len(nov_feats) == 58205 and len(sig_nov) == 58205, "11 月行数不符"

    # ---------- 断言 2：未知商品行必须整行填零，且不得有额外全零行 ----------
    unknown_mask = np.concatenate(unknown_blocks)
    zero_mask = (OUR_TR.abs().sum(axis=1) == 0).to_numpy()
    print(f"  [断言2] 各窗口未知商品行合计 {int(unknown_mask.sum()):,}；"
          f"实际全零行 {int(zero_mask.sum()):,}", flush=True)
    if not args.smoke_users:
        assert zero_mask[unknown_mask].all(), "存在未知商品行未被填零"
        extra = int(zero_mask.sum() - unknown_mask.sum())
        print(f"          未知商品行已全部填零 ✓；额外全零行 {extra}（窗口内无历史用户）", flush=True)
        assert zero_mask.sum() == unknown_mask.sum(), (
            f"出现 {extra} 行预期外的全零特征行，需检查用户覆盖逻辑"
        )

    OUR_TR.to_pickle(OUT / "our_feats_train.pkl")
    SIG_TR[SIGNAL_COLS].to_pickle(OUT / "signals_train.pkl")
    nov_feats.to_pickle(OUT / "our_feats_nov.pkl")
    sig_nov[SIGNAL_COLS].to_pickle(OUT / "signals_nov.pkl")
    KEYS_TR.to_pickle(OUT / "feature_keys_train.pkl")
    print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")
    print(f"  our_feats_train {OUR_TR.shape} | signals_train {SIG_TR[SIGNAL_COLS].shape}")
    print(f"  our_feats_nov   {nov_feats.shape} | signals_nov   {sig_nov[SIGNAL_COLS].shape}")


if __name__ == "__main__":
    main()
