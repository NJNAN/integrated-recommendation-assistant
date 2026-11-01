"""Verify every immutable file listed in an Oblivionis package manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"MISSING {entry['path']}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            failures.append(f"HASH {entry['path']} expected={entry['sha256']} actual={actual}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"FAIL: {len(failures)} manifest checks failed")
    print(f"PASS: {len(manifest['files'])} files match manifest.json")


if __name__ == "__main__":
    main()
