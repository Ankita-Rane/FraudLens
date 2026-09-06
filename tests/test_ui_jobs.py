from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import sys
import unittest

from src.ui_jobs import JobManager, JobSpec


class UiJobTests(unittest.TestCase):
    def manager(self, root: Path, code: str = "print('ok')") -> JobManager:
        spec = JobSpec(
            "SAFE_TEST", "Safe test", "Exercise the fixed command boundary.",
            (sys.executable, "-c", code),
        )
        return JobManager(root, registry={"SAFE_TEST": spec})

    def wait(self, manager: JobManager, job_id: str) -> dict:
        deadline = monotonic() + 10
        while monotonic() < deadline:
            result = manager.get(job_id)
            if result["terminal"]:
                return result
            sleep(0.02)
        self.fail("job did not complete")

    def test_allowlisted_job_persists_log_and_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self.manager(root)
            job = self.wait(manager, manager.create("SAFE_TEST")["job_id"])
            self.assertEqual(job["status"], "completed")
            self.assertEqual(manager.log_tail(job["job_id"]), ["ok"])
            self.assertTrue((root / job["manifest_path"]).is_file())
            restored = self.manager(root).get(job["job_id"])
            self.assertEqual(restored["status"], "completed")

    def test_unknown_job_and_parameters_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            manager = self.manager(Path(temporary))
            with self.assertRaises(ValueError):
                manager.create("SHELL_TEXT")
            with self.assertRaises(ValueError):
                manager.create("SAFE_TEST", {"command": "rm -rf anything"})

    def test_queued_job_can_be_cancelled_without_execution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self.manager(root, "print('must-not-run')")
            manager._lock.acquire()
            try:
                job = manager.create("SAFE_TEST")
                manager.cancel(job["job_id"])
            finally:
                manager._lock.release()
            result = self.wait(manager, job["job_id"])
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(result["stopping_reason"], "cancelled_before_start")
            self.assertEqual(manager.log_tail(job["job_id"]), [])

    def test_timeout_terminates_the_process_group(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = JobSpec(
                "TIMED", "Timed", "Exercise timeout enforcement.",
                (sys.executable, "-c", "import time; time.sleep(2)"),
                timeout_seconds=1,
            )
            manager = JobManager(root, registry={"TIMED": spec})
            result = self.wait(manager, manager.create("TIMED")["job_id"])
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["stopping_reason"].startswith("timeout_after_"))


if __name__ == "__main__":
    unittest.main()
