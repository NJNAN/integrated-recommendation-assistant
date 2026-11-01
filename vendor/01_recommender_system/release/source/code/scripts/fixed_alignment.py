# -*- coding: utf-8 -*-
"""修复版对齐工具：按 (_window, user_id, item_id) 键校验并对齐，禁止纯行号拼接。

背景
----
旧链路里 `make_submission_v15.complete_feature_frame` 会删掉 item 不在引擎
池里的候选行，下游却用 `pd.concat(axis=1)` 按 DataFrame 行号拼接，导致
290,581 行错位（实测：只有前 49 行键一致）。

本模块把"拼接"换成"先按键校验、再对齐"：任何一处不一致都会立刻抛错，
而不是静默错位。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["_window", "user_id", "item_id"]


def _keys(frame: pd.DataFrame) -> np.ndarray:
    return frame[KEY_COLS].astype(str).to_numpy()


def _check(tr: pd.DataFrame, keys: pd.DataFrame, label: str) -> None:
    if len(tr) != len(keys):
        raise ValueError(f"[{label}] 行数不符：标签 {len(tr)} vs 特征键 {len(keys)}")
    kt, kf = _keys(tr.reset_index(drop=True)), _keys(keys.reset_index(drop=True))
    if not (kt == kf).all():
        bad = (kt != kf).any(axis=1)
        raise ValueError(
            f"[{label}] 键顺序不一致：{int(bad.sum())} 行不符，"
            f"首个位置 {int(np.argmax(bad))} "
            f"（标签={kt[np.argmax(bad)].tolist()} 特征={kf[np.argmax(bad)].tolist()}）"
        )
    if pd.DataFrame(kt).duplicated().any():
        raise ValueError(f"[{label}] 键存在重复")


def _check_index(frame: pd.DataFrame, label: str) -> None:
    """位置对齐前提：index 的值必须恰好是 0..n-1（RangeIndex 或等值整数索引均可）。"""
    n = len(frame)
    if not np.array_equal(np.asarray(frame.index), np.arange(n)):
        head = list(frame.index[:5])
        raise ValueError(
            f"[{label}] index 不是 0..{n-1} 的位置索引（前 5 个：{head}），"
            "无法按键校验后按位置放置"
        )


def load_aligned_train(
    tr: pd.DataFrame, in_dir: Path, signal_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 v120 的 5 窗口特征块按 (_window, user_id, item_id) 对齐到 tr 的行序。

    返回 (our_feats, signals)，两者 index 已设为 tr.index。
    """
    keys = pd.read_pickle(in_dir / "feature_keys_train.pkl")
    ours = pd.read_pickle(in_dir / "our_feats_train.pkl")
    sig = pd.read_pickle(in_dir / "signals_train.pkl")

    if not (len(tr) == len(ours) == len(sig)):
        raise ValueError(
            f"[train] 行数不符：标签 {len(tr)} / our_feats {len(ours)} / signals {len(sig)}"
        )
    _check(tr, keys, "train")
    _check_index(ours, "our_feats_train")
    _check_index(sig, "signals_train")

    ours = ours.astype(float)
    sig = sig[signal_cols].astype(float)
    ours.index = tr.index
    sig.index = tr.index
    print(f"  [对齐] train 键校验通过：{len(tr):,} 行，唯一键 {len(tr):,} 个", flush=True)
    return ours, sig


def load_aligned_nov(
    full: pd.DataFrame, in_dir: Path, cand: pd.DataFrame, signal_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """11 月：按候选顺序校验并对其。"""
    ours = pd.read_pickle(in_dir / "our_feats_nov.pkl")
    sig = pd.read_pickle(in_dir / "signals_nov.pkl")

    for name, frame in (("full", full), ("our_feats_nov", ours), ("signals_nov", sig)):
        if len(frame) != len(cand):
            raise ValueError(f"[nov] {name} 行数 {len(frame)} != 候选集 {len(cand)}")

    ck = (cand.user_id.astype(str) + "|" + cand.item_id.astype(str)).to_numpy()
    fk = (full.user_id.astype(str) + "|" + full.item_id.astype(str)).to_numpy()
    if not (ck == fk).all():
        bad = ck != fk
        raise ValueError(f"[nov] full.pkl 键与候选集不一致：{int(bad.sum())} 行不符")

    _check_index(ours, "our_feats_nov")
    _check_index(sig, "signals_nov")

    ours = ours.astype(float)
    sig = sig[signal_cols].astype(float)
    ours.index = full.index
    sig.index = full.index
    print(f"  [对齐] nov 键校验通过：{len(cand):,} 行", flush=True)
    return ours, sig
