"""Build checksums for the files included in the public research release."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility"
HASH_FILE = OUTPUT / "artifact_sha256.csv"
SUMMARY_FILE = OUTPUT / "manifest_summary.json"
EXCLUDED_PARTS = {".git", ".venv", ".matplotlib", "__pycache__", ".pytest_cache"}
EXCLUDED_FILES = {HASH_FILE.resolve(), SUMMARY_FILE.resolve()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.resolve() in EXCLUDED_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in public_files()
    ]
    with HASH_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "completed",
        "hash_algorithm": "SHA-256",
        "artifacts_hashed": len(rows),
        "hash_file": HASH_FILE.name,
        "environment_file": "environment.json",
        "resource_file": "runtime_resources.csv",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
