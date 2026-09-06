"""Integrity checks for immutable thesis input artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import json


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def audit_input_artifacts(project_root: Path) -> dict[str, Any]:
    """Validate presence, location, file count, and declared SHA-256 hashes."""
    manifest_path = project_root / "datasets" / "input_artifacts.json"
    if not manifest_path.exists():
        return {
            "status": "failed",
            "manifest": str(manifest_path),
            "checks": [],
            "errors": ["Input artifact manifest is missing."],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    dataset_root = (project_root / "datasets").resolve()

    for item in manifest.get("artifacts", []):
        path = (project_root / item["path"]).resolve()
        inside_dataset_root = path == dataset_root or dataset_root in path.parents
        exists = path.exists()
        check: dict[str, Any] = {
            "id": item["id"],
            "path": str(path.relative_to(project_root)) if inside_dataset_root else str(path),
            "exists": exists,
            "inside_datasets": inside_dataset_root,
            "required": bool(item.get("required", False)),
            "passed": exists and inside_dataset_root,
        }
        if path.is_dir():
            file_count = sum(1 for child in path.rglob("*") if child.is_file())
            check["file_count"] = file_count
            minimum_files = int(item.get("minimum_files", 0))
            check["minimum_files"] = minimum_files
            check["passed"] = check["passed"] and file_count >= minimum_files
        elif exists and item.get("sha256"):
            actual_hash = file_sha256(path)
            check["expected_sha256"] = item["sha256"]
            check["actual_sha256"] = actual_hash
            check["passed"] = check["passed"] and actual_hash == item["sha256"]

        if not check["passed"] and check["required"]:
            errors.append(f"Required input failed integrity check: {item['id']}")
        checks.append(check)

    policy_registry_path = (
        project_root / "datasets" / "policy_sources" / "policy_source_registry.json"
    )
    if policy_registry_path.exists():
        policy_registry = json.loads(policy_registry_path.read_text(encoding="utf-8"))
        policy_directory = policy_registry_path.parent / "source"
        for source in policy_registry.get("sources", []):
            path = policy_directory / source["local_filename"]
            exists = path.is_file()
            actual_hash = file_sha256(path) if exists else None
            expected_hash = source.get("sha256")
            passed = exists and bool(expected_hash) and actual_hash == expected_hash
            checks.append({
                "id": f"policy:{source['local_filename']}",
                "path": str(path.relative_to(project_root)),
                "exists": exists,
                "inside_datasets": True,
                "required": True,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "passed": passed,
            })
            if not passed:
                errors.append(
                    f"Frozen policy source failed integrity check: {source['local_filename']}"
                )

    return {
        "status": "passed" if not errors else "failed",
        "manifest": str(manifest_path.relative_to(project_root)),
        "checks": checks,
        "errors": errors,
    }
