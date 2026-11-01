"""Run the complete training chain for Oblivionis(dd2poolseed).

The script is intentionally an orchestrator: it preserves the audited versioned
experiments while providing one stable command for a from-source rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


EXPECTED_SHA256 = "123816780ae366ee5e9bea43cd69d4188bc55fe8c12cd67c126b9923a25518e3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def run_script(project_root: Path, name: str) -> None:
    script = project_root / "scripts" / name
    require(script, "training script")
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to the code directory",
    )
    parser.add_argument(
        "--force-prerequisites",
        action="store_true",
        help="rebuild v65 cache and v67 binary even when their outputs exist",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    require(root / "data" / "train.csv", "competition training data")
    require(root / "data" / "sample_candidates.csv", "candidate set")
    require(root / "models" / "competition_v11_season" / "engine.pkl", "baseline engine")
    require(root / "scripts" / "experiment_v94_O2_H2.py", "v94 module dependency")

    # 预检：七步链的完整传递 import 闭包。
    # 缺任何一个都会在链跑到中途才报 ModuleNotFoundError —— 这里一次性全查。
    for name in (
        "scripts/experiment_v65_capacity.py",
        "scripts/experiment_v67_gauc_objective.py",
        "scripts/experiment_v94_O2_H2.py",
        "scripts/experiment_v95_O2_H2_fixed.py",
        "scripts/experiment_v102_H2_variants.py",
        "scripts/experiment_v104_smallcap.py",
        "scripts/experiment_v105_smallpool.py",
        "scripts/experiment_v106_headfocus.py",
        "scripts/train_competition.py",
        "scripts/train_competition_v9.py",
        "scripts/train_competition_v15.py",
        "scripts/make_submission_v11_season.py",
        "scripts/make_submission_v15.py",
        "scripts/signals_third_family.py",
        "baseline/config.py",
        "baseline/bpr.py",
        "baseline/data.py",
        "baseline/engine.py",
        "baseline/competition.py",
        "baseline/features_v15.py",
        "baseline/seasonal_features.py",
    ):
        require(root / name, "chain module (import closure)")

    # 链外输入：v104/v105/v106 的 sources 字典里 ours 指向 v47 的产物。
    # 它是"项目自有模型"的冻结排序，不由本链产出，必须随根提供。
    require(
        root / "outputs/competition_v47_right_sweep/submission_v47_blend65_v36_w35.csv",
        "chain external input (ours / v47 blend)",
    )

    # 链外输入：v104/v105/v106 的 sources 字典里 H94 指向 v94 的产物，
    # 但七步链是从 v95 起跑的（v95 是 v94 的修复版），v94 的产物必须事先就位。
    # 注：H95 由本链的 v95 步骤重新产出，此处一并预置作为兜底。
    require(
        root / "outputs/competition_v94_O2_H2/submission_v94_H2.csv",
        "chain external input (H94 / v94 output)",
    )
    require(
        root / "outputs/competition_v95_O2_H2_fixed/submission_v95_H2.csv",
        "chain external input (H95 / v95 output fallback)",
    )

    cache = root / "outputs" / "competition_v65_capacity" / "cache"
    binary = (
        root
        / "outputs"
        / "competition_v67_gauc_objective"
        / "submission_v67_binary_v9blend.csv"
    )
    if args.force_prerequisites or not cache.is_dir():
        run_script(root, "experiment_v65_capacity.py")
    else:
        print(f"reuse prerequisite cache: {cache}")
    if args.force_prerequisites or not binary.is_file():
        run_script(root, "experiment_v67_gauc_objective.py")
    else:
        print(f"reuse prerequisite binary: {binary}")

    for script in (
        "experiment_v95_O2_H2_fixed.py",
        "experiment_v102_H2_variants.py",
        "experiment_v104_smallcap.py",
        "experiment_v105_smallpool.py",
        "experiment_v106_headfocus.py",
    ):
        run_script(root, script)

    final = (
        root
        / "outputs"
        / "competition_v106_headfocus"
        / "submission_DD2_pool_seed.csv"
    )
    require(final, "final submission")
    digest = sha256(final)
    print(f"\nfinal: {final}")
    print(f"sha256: {digest}")
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            "training completed, but the final hash differs from the locked artifact; "
            "check dependency and library versions"
        )
    print("PASS: Oblivionis(dd2poolseed) exact artifact reproduced")


if __name__ == "__main__":
    main()
