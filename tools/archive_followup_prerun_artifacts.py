"""Hash-verify and archive replaceable follow-up detector/panel artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ID = "FOLLOWUP-PREFRESH-20260902-01"
ARCHIVE_ROOT = ROOT / "artifacts" / "old_evidence" / ARCHIVE_ID


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory() -> list[Path]:
    files = set(path for path in (ROOT / "artifacts" / "followup_detectors").rglob("*") if path.is_file())
    for relative in (
        "datasets/ulb/followup_transaction_samples.csv",
        "datasets/sparkov/followup_transaction_samples.csv",
        "datasets/evaluation/followup_retrieval/followup_evaluation_labels.csv",
        "datasets/evaluation/followup_retrieval/followup_panel_manifest.json",
        "datasets/evaluation/followup_retrieval/panel_feasibility.json",
    ):
        path = ROOT / relative
        if path.is_file():
            files.add(path)
    return sorted(files)


def manifest() -> dict:
    files = inventory()
    return {
        "schema_version": "1.0",
        "archive_id": ARCHIVE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "replaceable pre-fresh follow-up detector, panel, attribution and feasibility artifacts",
        "raw_datasets_moved": False,
        "retrieval_diagnostic_moved": False,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [{
            "source_relative_path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        } for path in files],
    }


def apply(payload: dict) -> Path:
    manifest_path = ARCHIVE_ROOT / "archive_manifest.json"
    if ARCHIVE_ROOT.exists():
        raise FileExistsError(f"Archive already exists: {ARCHIVE_ROOT}")
    snapshot = ARCHIVE_ROOT / "repository_snapshot"
    for row in payload["files"]:
        source = ROOT / row["source_relative_path"]
        if digest(source) != row["sha256"]:
            raise RuntimeError(f"Source changed after inventory: {source}")
    snapshot.mkdir(parents=True)
    for row in payload["files"]:
        source = ROOT / row["source_relative_path"]
        destination = snapshot / row["source_relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        if digest(destination) != row["sha256"]:
            raise RuntimeError(f"Destination hash mismatch: {destination}")
    payload["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "archived_hash_verified"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload = manifest()
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    output = apply(payload)
    print(json.dumps({
        "status": "archived_hash_verified",
        "manifest": str(output.relative_to(ROOT)),
        "manifest_sha256": digest(output),
        "file_count": payload["file_count"],
        "total_bytes": payload["total_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
