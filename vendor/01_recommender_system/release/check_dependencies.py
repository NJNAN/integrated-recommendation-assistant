"""Verify every external dependency listed in dependency_inventory.json.

The second-layer reproduction needs ~525 MB of inputs that are not copied into
this package.  Run this before `retrain_oblivionis.py` to find out whether the
project root actually contains them, and whether they are the ones the locked
result was built from.

Usage:
    python check_dependencies.py --project-root D:/作业/行业工程实践/code
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="path to the code directory that holds data/, models/, outputs/",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).resolve().parent / "dependency_inventory.json",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="only check the 4 largest groups' presence, skip SHA256 (faster)",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total_bytes = 0

    for entry in inventory:
        path = root / entry["path"]
        rel = entry["path"]
        if not path.is_file():
            groups[entry["role"]].append((rel, "MISSING"))
            continue
        total_bytes += path.stat().st_size
        if args.quick:
            groups[entry["role"]].append((rel, "ok"))
            continue
        actual = sha256(path)
        groups[entry["role"]].append(
            (rel, "ok" if actual == entry["sha256"] else f"HASH {actual[:16]} != {entry['sha256'][:16]}")
        )

    bad: list[str] = []
    print(f"project root: {root}")
    print(f"inventory   : {len(inventory)} entries\n")

    for role in sorted(groups, key=lambda r: -len(groups[r])):
        rows = groups[role]
        ok = sum(1 for _, s in rows if s == "ok")
        size = sum(
            (root / e["path"]).stat().st_size
            for e in inventory
            if e["role"] == role and (root / e["path"]).is_file()
        )
        mark = "PASS" if ok == len(rows) else "FAIL"
        print(f"[{mark}] {role:24s} {ok}/{len(rows)} 项   {size / 1e6:8.1f} MB")
        for rel, status in rows:
            if status != "ok":
                print(f"         {status:34s} {rel}")
                bad.append(rel)

    print(f"\n已就位合计: {total_bytes / 1e6:.1f} MB")
    if bad:
        raise SystemExit(f"\nFAIL: {len(bad)} 项缺失或哈希不符")
    print("PASS: 全部依赖已就位且哈希吻合")


if __name__ == "__main__":
    main()
