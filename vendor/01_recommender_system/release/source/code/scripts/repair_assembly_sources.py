# -*- coding: utf-8 -*-
"""把 v121~v124 的装配源 H94/H95 指向 v120 修复版。"""
from pathlib import Path

S = Path("D:/作业/行业工程实践/code/scripts")

OLD_H94 = '"H94": ROOT / "outputs/competition_v94_O2_H2/submission_v94_H2.csv"'
NEW_H94 = '"H94": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H94fixed.csv"'
OLD_H95 = '"H95": ROOT / "outputs/competition_v95_O2_H2_fixed/submission_v95_H2.csv"'
NEW_H95 = '"H95": ROOT / "outputs/competition_v120_H2_fixed/submission_v120_H95fixed.csv"'

for name in ("experiment_v121_H2_variants_fixed.py",
             "experiment_v122_smallcap_fixed.py",
             "experiment_v123_smallpool_fixed.py",
             "experiment_v124_headfocus_fixed.py"):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n94, n95 = t.count(OLD_H94), t.count(OLD_H95)
    t = t.replace(OLD_H94, NEW_H94).replace(OLD_H95, NEW_H95)
    p.write_text(t, encoding="utf-8")
    print(f"[{name}] H94 替换 {n94} 处 | H95 替换 {n95} 处")
