# -*- coding: utf-8 -*-
"""从 v102/v104/v105/v106 机械生成修复版 v121~v124。

只改三处：
  1. IN  指向 v120 修复版特征目录
  2. OUT 指向新的输出目录
  3. 把 Xtr/Xfu 的纯行号 pd.concat 换成 fixed_alignment 的键校验对齐
逻辑（超参数、配方、emit 顺序）一律不动。
"""
from pathlib import Path

ROOT = Path("D:/作业/行业工程实践/code")
S = ROOT / "scripts"

JOBS = [
    ("experiment_v102_H2_variants.py", "experiment_v121_H2_variants_fixed.py",
     "outputs/competition_v102_H2_variants", "outputs/competition_v121_H2_variants_fixed",
     "submission_v102_", "submission_v121_"),
    ("experiment_v104_smallcap.py", "experiment_v122_smallcap_fixed.py",
     "outputs/competition_v104_smallcap", "outputs/competition_v122_smallcap_fixed",
     "submission_v104_", "submission_v122_"),
    ("experiment_v105_smallpool.py", "experiment_v123_smallpool_fixed.py",
     "outputs/competition_v105_smallpool", "outputs/competition_v123_smallpool_fixed",
     "submission_v105_", "submission_v123_"),
    ("experiment_v106_headfocus.py", "experiment_v124_headfocus_fixed.py",
     "outputs/competition_v106_headfocus", "outputs/competition_v124_headfocus_fixed",
     "submission_v106_", "submission_v124_"),
]

# v102 用的是显式变量名，单独处理
BLOCK_V102_OLD = '''    OUR_TR = pd.read_pickle(IN / "our_feats_train.pkl")
    SIG_TR = pd.read_pickle(IN / "signals_train.pkl")
    nov_ours = pd.read_pickle(IN / "our_feats_nov.pkl")
    sig_nov = pd.read_pickle(IN / "signals_nov.pkl")
    Xtr = pd.concat([tr[base_feats].astype(float), OUR_TR.astype(float),
                     SIG_TR[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float), nov_ours.astype(float),
                     sig_nov[v94.SIGNAL_COLS]], axis=1)'''

BLOCK_V102_NEW = '''    # [修复] 按键校验对齐，禁止纯行号拼接
    OUR_TR, SIG_TR = load_aligned_train(tr, IN, v94.SIGNAL_COLS)
    nov_ours, sig_nov = load_aligned_nov(full, IN, cand_pairs, v94.SIGNAL_COLS)
    Xtr = pd.concat([tr[base_feats].astype(float), OUR_TR,
                     SIG_TR[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float), nov_ours,
                     sig_nov[v94.SIGNAL_COLS]], axis=1)'''

BLOCK_OLD = '''    Xtr = pd.concat([tr[base_feats].astype(float),
                     pd.read_pickle(IN / "our_feats_train.pkl").astype(float),
                     pd.read_pickle(IN / "signals_train.pkl")[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float),
                     pd.read_pickle(IN / "our_feats_nov.pkl").astype(float),
                     pd.read_pickle(IN / "signals_nov.pkl")[v94.SIGNAL_COLS]], axis=1)'''

BLOCK_NEW = '''    # [修复] 按键校验对齐，禁止纯行号拼接
    _ours_tr, _sig_tr = load_aligned_train(tr, IN, v94.SIGNAL_COLS)
    _ours_fu, _sig_fu = load_aligned_nov(full, IN, cand_pairs, v94.SIGNAL_COLS)
    Xtr = pd.concat([tr[base_feats].astype(float), _ours_tr,
                     _sig_tr[v94.SIGNAL_COLS]], axis=1)
    Xfu = pd.concat([full[base_feats].astype(float), _ours_fu,
                     _sig_fu[v94.SIGNAL_COLS]], axis=1)'''

for src_name, dst_name, old_out, new_out, old_pref, new_pref in JOBS:
    text = (S / src_name).read_text(encoding="utf-8")
    n = {}

    def sub(a, b, label):
        c = text.count(a)
        n[label] = c
        return text.replace(a, b)

    text = sub('IN = ROOT / "outputs/competition_v95_O2_H2_fixed"',
               'IN = ROOT / "outputs/competition_v120_H2_fixed"', "IN")
    text = sub(f'OUT = ROOT / "{old_out}"', f'OUT = ROOT / "{new_out}"', "OUT")
    text = sub(old_pref, new_pref, "文件名前缀")
    # 插入 import
    marker = "from scripts.train_competition import file_sha256"
    assert marker in text, f"{src_name}: 找不到 import 锚点"
    if "fixed_alignment" not in text:
        text = text.replace(
            marker,
            "from scripts.fixed_alignment import (  # noqa: E402\n"
            "    load_aligned_nov, load_aligned_train,\n)\n" + marker, 1)
    # 替换拼接块
    if BLOCK_V102_OLD in text:
        text = text.replace(BLOCK_V102_OLD, BLOCK_V102_NEW)
        n["拼接块"] = 1
    elif BLOCK_OLD in text:
        text = text.replace(BLOCK_OLD, BLOCK_NEW)
        n["拼接块"] = 1
    else:
        n["拼接块"] = 0

    (S / dst_name).write_text(text, encoding="utf-8")
    print(f"[生成] {dst_name}")
    for k, v in n.items():
        print(f"    {k:10s} 替换 {v} 处")
