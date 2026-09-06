"""Persistent, allowlisted subprocess jobs for the local research console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from contextlib import contextmanager
from subprocess import PIPE, STDOUT, Popen, TimeoutExpired
from threading import Lock, Thread
from typing import Any, Mapping, Sequence

import json
import logging
import os
import signal
import sqlite3
import sys
import uuid


LOGGER = logging.getLogger(__name__)
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@dataclass(frozen=True)
class JobSpec:
    job_type: str
    title: str
    purpose: str
    argv: tuple[str, ...]
    output_paths: tuple[str, ...] = ()
    compute_heavy: bool = False
    external_write: bool = False
    timeout_seconds: int = 7_200

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("argv")
        return payload


def default_job_registry(project_root: Path) -> dict[str, JobSpec]:
    py = sys.executable
    return {
        "INPUT_AUDIT": JobSpec(
            "INPUT_AUDIT", "Audit registered inputs",
            "Validate registered paths, counts and hashes without changing evidence.",
            (py, "tools/audit_input_artifacts.py"),
        ),
        "SMOKE_TESTS": JobSpec(
            "SMOKE_TESTS", "Run detector smoke tests",
            "Catch schema and dependency failures before expensive model builds.",
            (py, "tests/smoke_test.py"), compute_heavy=True,
        ),
        "ULB_NOTEBOOK_BUILD": JobSpec(
            "ULB_NOTEBOOK_BUILD", "Build ULB detector",
            "Execute the independent ULB EDA and modelling notebook.",
            (py, "tools/execute_notebook.py", "notebooks/ULB_EDA_MODELING.ipynb",
             "--working-directory", ".", "--timeout", "7200"),
            ("artifacts/ulb/run_manifest.json", "artifacts/ulb/test_metrics.csv"),
            compute_heavy=True, timeout_seconds=7_500,
        ),
        "SPARKOV_NOTEBOOK_BUILD": JobSpec(
            "SPARKOV_NOTEBOOK_BUILD", "Build Sparkov detector",
            "Execute the independent Sparkov EDA and modelling notebook.",
            (py, "tools/execute_notebook.py", "notebooks/SPARKOV_EDA_MODELING.ipynb",
             "--working-directory", ".", "--timeout", "7200"),
            ("artifacts/sparkov/run_manifest.json", "artifacts/sparkov/test_metrics.csv"),
            compute_heavy=True, timeout_seconds=7_500,
        ),
        "BUILD_RUNTIME_EVIDENCE": JobSpec(
            "BUILD_RUNTIME_EVIDENCE", "Build runtime evidence",
            "Refresh source-labelled runtime panels, profiles and unified evidence.",
            (py, "tools/run_ui_workflow_step.py", "runtime-evidence"),
            ("artifacts/unified/unified_model_evidence.json",), compute_heavy=True,
        ),
        "BUILD_RUNTIME_ATTRIBUTIONS": JobSpec(
            "BUILD_RUNTIME_ATTRIBUTIONS", "Build runtime attributions",
            "Generate model-faithful local attributions for runtime samples.",
            (py, "tools/build_runtime_attributions.py", "--panel", "runtime"),
            ("artifacts/ulb/runtime_sample_attributions.json",
             "artifacts/sparkov/runtime_sample_attributions.json"),
            compute_heavy=True,
        ),
        "RUN_REPOSITORY_TESTS": JobSpec(
            "RUN_REPOSITORY_TESTS", "Run repository tests",
            "Verify scientific, API, Jira and presentation contracts.",
            (py, "-m", "unittest", "discover", "-s", "tests", "-v"),
            compute_heavy=True,
        ),
        "BUILD_SELECTED_HTML_REPORT": JobSpec(
            "BUILD_SELECTED_HTML_REPORT", "Build primary HTML report",
            "Generate the primary light-theme presentation as the final projection.",
            (py, "tools/build_thesis_results_showcase.py"),
            ("thesis_results_showcase.html",),
        ),
    }


class JobManager:
    """Single-worker persistent manager; no caller-provided executable is accepted."""

    def __init__(
        self,
        project_root: Path,
        *,
        registry: Mapping[str, JobSpec] | None = None,
        database_path: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.runtime_root = self.project_root / "artifacts" / "ui_jobs"
        self.log_root = self.project_root / "log" / "jobs"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path or self.runtime_root / "jobs.sqlite"
        self.registry = dict(registry or default_job_registry(self.project_root))
        self._lock = Lock()
        self._active_processes: dict[str, Popen[str]] = {}
        self._initialize_database()
        self._reconcile_interrupted()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _database(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._database() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    started_at_utc TEXT,
                    completed_at_utc TEXT,
                    return_code INTEGER,
                    stopping_reason TEXT,
                    log_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    process_id INTEGER
                )
            """)

    def _reconcile_interrupted(self) -> None:
        now = _utc_now()
        with self._database() as connection:
            connection.execute(
                """UPDATE jobs SET status='interrupted', completed_at_utc=?,
                   stopping_reason='api_restart_or_worker_loss'
                   WHERE status='running'""",
                (now,),
            )

    def list_specs(self) -> list[dict[str, Any]]:
        return [self.registry[key].public() for key in sorted(self.registry)]

    def create(self, job_type: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if job_type not in self.registry:
            raise ValueError("Unknown or non-allowlisted job type.")
        parameters = dict(parameters or {})
        if parameters:
            raise ValueError("This job type accepts no caller-supplied parameters.")
        spec = self.registry[job_type]
        if spec.external_write:
            raise ValueError("External-write jobs require their dedicated gated API.")
        job_id = str(uuid.uuid4())
        directory = self.runtime_root / job_id
        directory.mkdir(parents=False, exist_ok=False)
        log_path = self.log_root / f"{job_id}.log"
        manifest_path = directory / "manifest.json"
        created = _utc_now()
        with self._database() as connection:
            connection.execute(
                """INSERT INTO jobs VALUES (?, ?, ?, ?, 'queued', ?, ?, NULL, NULL,
                   NULL, NULL, ?, ?, NULL)""",
                (job_id, spec.job_type, spec.title, spec.purpose,
                 json.dumps(parameters, sort_keys=True), created,
                 str(log_path.relative_to(self.project_root)),
                 str(manifest_path.relative_to(self.project_root))),
            )
        self._event(job_id, "queued", {"job_type": job_type})
        Thread(target=self._run, args=(job_id,), daemon=True).start()
        return self.get(job_id)

    def _event(self, job_id: str, event: str, detail: Mapping[str, Any]) -> None:
        path = self.runtime_root / job_id / "events.jsonl"
        record = {"timestamp_utc": _utc_now(), "event": event, **dict(detail)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        spec = self.registry[job["job_type"]]
        with self._lock:
            job = self.get(job_id)
            if job["status"] == "cancelling":
                completed = _utc_now()
                with self._database() as connection:
                    connection.execute(
                        """UPDATE jobs SET status='cancelled', completed_at_utc=?,
                           stopping_reason='cancelled_before_start', process_id=NULL
                           WHERE job_id=?""",
                        (completed, job_id),
                    )
                self._event(job_id, "finished", {
                    "status": "cancelled", "return_code": None,
                    "reason": "cancelled_before_start",
                })
                self._write_manifest(job_id)
                return
            started = _utc_now()
            with self._database() as connection:
                connection.execute(
                    "UPDATE jobs SET status='running', started_at_utc=? WHERE job_id=?",
                    (started, job_id),
                )
            self._event(job_id, "started", {"argv_id": spec.job_type})
            log_path = self.project_root / job["log_path"]
            global_log = self.project_root / "log" / "application.log"
            global_log.parent.mkdir(parents=True, exist_ok=True)
            status = "failed"
            reason = "non_zero_exit"
            return_code: int | None = None
            process: Popen[str] | None = None
            try:
                process = Popen(
                    spec.argv,
                    cwd=self.project_root,
                    stdout=PIPE,
                    stderr=STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self._active_processes[job_id] = process
                with self._database() as connection:
                    connection.execute(
                        "UPDATE jobs SET process_id=? WHERE job_id=?",
                        (process.pid, job_id),
                    )
                with log_path.open("a", encoding="utf-8") as local, global_log.open(
                    "a", encoding="utf-8"
                ) as global_handle:
                    assert process.stdout is not None
                    def pump_output() -> None:
                        assert process is not None and process.stdout is not None
                        with process.stdout:
                            for line in process.stdout:
                                local.write(line)
                                local.flush()
                                global_handle.write(
                                    f"{_utc_now()} | INFO | ui_job | job_id={job_id} " + line
                                )
                                global_handle.flush()

                    reader = Thread(target=pump_output, daemon=True)
                    reader.start()
                    try:
                        return_code = process.wait(timeout=spec.timeout_seconds)
                    except TimeoutExpired:
                        reason = f"timeout_after_{spec.timeout_seconds}_seconds"
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                        return_code = process.returncode
                    reader.join(timeout=10)
                current = self.get(job_id)["status"]
                if current == "cancelling":
                    status, reason = "cancelled", "user_cancelled"
                elif return_code == 0 and not reason.startswith("timeout_after_"):
                    status, reason = "completed", "process_exit_zero"
            except Exception as error:  # subprocess boundary must preserve the error
                reason = f"{type(error).__name__}: {error}"[:500]
                LOGGER.exception("ui_job_failed job_id=%s job_type=%s", job_id, spec.job_type)
                if process is not None and process.poll() is None:
                    process.terminate()
            finally:
                self._active_processes.pop(job_id, None)
                completed = _utc_now()
                with self._database() as connection:
                    connection.execute(
                        """UPDATE jobs SET status=?, completed_at_utc=?, return_code=?,
                           stopping_reason=?, process_id=NULL WHERE job_id=?""",
                        (status, completed, return_code, reason, job_id),
                    )
                self._event(job_id, "finished", {
                    "status": status, "return_code": return_code, "reason": reason,
                })
                self._write_manifest(job_id)

    def _write_manifest(self, job_id: str) -> None:
        job = self.get(job_id)
        spec = self.registry[job["job_type"]]
        outputs = []
        for relative in spec.output_paths:
            path = self.project_root / relative
            outputs.append({
                "path": relative,
                "exists": path.exists(),
                "sha256": _sha256(path),
            })
        payload = {
            "schema_version": "1.0",
            "actor": "local_ui",
            "command_registry_version": "1.0",
            **job,
            "outputs": outputs,
        }
        path = self.project_root / job["manifest_path"]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in TERMINAL_STATUSES:
            return job
        with self._database() as connection:
            connection.execute(
                "UPDATE jobs SET status='cancelling', stopping_reason='cancel_requested' WHERE job_id=?",
                (job_id,),
            )
        process = self._active_processes.get(job_id)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._event(job_id, "cancel_requested", {})
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._database() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        payload = dict(row)
        payload["parameters"] = json.loads(payload.pop("parameters_json"))
        payload["terminal"] = payload["status"] in TERMINAL_STATUSES
        payload["events"] = self.events(job_id, limit=100)
        return payload

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500.")
        with self._database() as connection:
            rows = connection.execute(
                "SELECT job_id FROM jobs ORDER BY created_at_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get(str(row["job_id"])) for row in rows]

    def events(self, job_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        path = self.runtime_root / job_id / "events.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def log_tail(self, job_id: str, *, lines: int = 120) -> list[str]:
        job = self.get(job_id)
        path = self.project_root / job["log_path"]
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
