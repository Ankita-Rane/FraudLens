"""Execute a notebook in place with progress messages and strict error handling."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=7_200)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    working_directory = args.working_directory.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)

    def on_cell_start(cell, cell_index, **_) -> None:
        if cell.cell_type == "code":
            first_line = "".join(cell.source).strip().splitlines()
            label = first_line[0][:100] if first_line else "<empty>"
            print(f"[{notebook_path.name}] code cell {cell_index}: {label}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        startup_timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
        allow_errors=False,
        on_cell_start=on_cell_start,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Executed and saved {notebook_path}", flush=True)


if __name__ == "__main__":
    main()
