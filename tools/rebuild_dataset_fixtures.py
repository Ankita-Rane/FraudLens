"""Reconstruct local dataset fixtures from raw downloads and a locked selector manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset_fixture_support import (
    reconstruct_fixture,
    sha256_file,
    write_fixture,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "datasets" / "evaluation" / "dataset_fixture_selection_manifest.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported fixture manifest schema version.")
    if manifest.get("contains_transaction_values") is not False:
        raise ValueError("Fixture manifest transaction-value boundary is missing.")

    for record in manifest["datasets"].values():
        path = PROJECT_ROOT / record["source_path"]
        actual = sha256_file(path)
        if actual != record["source_sha256"]:
            raise ValueError(f"Source digest mismatch for {path}.")

    for specification in manifest["fixtures"]:
        for kind in ("pipeline", "metrics"):
            path = PROJECT_ROOT / specification[f"{kind}_path"]
            actual = sha256_file(path)
            if actual != specification[f"{kind}_sha256"]:
                raise ValueError(f"{kind} digest mismatch for {path}.")

    output_root = args.output_root.expanduser().resolve()
    results = []
    for specification in manifest["fixtures"]:
        output = output_root / specification["path"]
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite.")
        frame = reconstruct_fixture(specification)
        if len(frame) != specification["expected_rows"]:
            raise ValueError(f"Row-count mismatch for {specification['path']}.")
        write_fixture(frame, output, specification)
        actual_hash = sha256_file(output)
        expected_hash = specification["expected_sha256"]
        if args.verify and actual_hash != expected_hash:
            raise ValueError(
                f"Fixture digest mismatch for {specification['path']}: "
                f"{actual_hash} != {expected_hash}"
            )
        results.append({
            "path": str(output),
            "rows": len(frame),
            "sha256": actual_hash,
            "verified": actual_hash == expected_hash,
        })
    print(json.dumps({
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "fixtures": results,
        "all_verified": all(item["verified"] for item in results),
    }, indent=2))


if __name__ == "__main__":
    main()
