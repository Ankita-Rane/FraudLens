"""Hash the archived scope/benchmark artifacts without modifying their contents."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets" / "evaluation" / "followup_retrieval" / "old"
OUTPUT = ARCHIVE / "archive_manifest.json"
TARGETS = (
    ROOT / "datasets" / "policy_sources" / "old" / "policy_source_registry_v1_1.json",
    *sorted(path for path in ARCHIVE.glob("*.json") if path.name != OUTPUT.name),
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite archive manifest: {OUTPUT}")
    files = [{
        "archive_path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "status": "superseded_provenance_not_valid_for_new_protocol",
    } for path in TARGETS]
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_status": "hash_verified_superseded_scope_artifacts",
        "replacement_policy": "Archived bytes are immutable; current canonical artifacts receive new hashes.",
        "file_count": len(files),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "sha256": digest(OUTPUT), "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
