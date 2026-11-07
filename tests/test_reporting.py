"""Regression tests for setting-level diagnostic log collection."""

import json
from pathlib import Path
import tempfile
import unittest

from swebench_docker.constants import (
    INSTALL_FAIL,
    RESET_FAILED,
    TESTS_ERROR,
    TESTS_TIMEOUT,
)
from swebench_docker.swebench_utils import get_model_report


class ModelReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="testgeneval-report-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.predictions = self.root / "predictions.jsonl"
        self.model = "fixture-model"

    def write_predictions(self, rows):
        self.predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def write_log(self, sample_id, text, setting=None):
        suffix = f".{setting}" if setting else ""
        path = self.log_dir / f"{sample_id}.{self.model}{suffix}.eval.log"
        path.write_text(text, encoding="utf-8")

    def get_report(self):
        return get_model_report(
            self.model, str(self.predictions), str(self.log_dir)
        )

    def test_setting_logs_collect_different_failures_once_per_sample(self):
        self.write_predictions([
            {"id": "sample", "preds": {
                "full": ["test"], "first": ["test"],
                "last": ["test"], "extra": ["test"],
            }},
        ])
        self.write_log("sample", INSTALL_FAIL + "\n" + RESET_FAILED, "full")
        self.write_log("sample", TESTS_ERROR + "\nMutationTimeout", "first")
        self.write_log("sample", TESTS_TIMEOUT + "\nMutationTimeout", "last")
        self.write_log("sample", TESTS_ERROR + "\n" + INSTALL_FAIL, "extra")

        report = self.get_report()

        self.assertEqual(report["no_generation"], [])
        for category in (
            "generated", "with_logs", "install_fail", "reset_failed",
            "test_errored", "test_timeout", "mutation_timeout",
        ):
            self.assertEqual(report[category], ["sample"], category)

    def test_missing_logs_are_not_counted_as_evaluated(self):
        self.write_predictions([
            {"id": "missing", "preds": {"full": ["test"]}},
            {"id": "partial", "preds": {"full": ["test"], "first": ["test"]}},
        ])
        self.write_log("partial", "A setting log exists.\n", "first")

        report = self.get_report()

        self.assertEqual(report["generated"], ["missing", "partial"])
        self.assertEqual(report["with_logs"], ["partial"])
        self.assertEqual(report["test_errored"], [])

    def test_baseline_aggregate_filename_remains_supported(self):
        self.model = "baseline"
        self.write_predictions([{"id": "baseline-sample"}])
        self.write_log("baseline-sample", TESTS_TIMEOUT + "\nMutationTimeout")

        report = self.get_report()

        self.assertEqual(report["with_logs"], ["baseline-sample"])
        self.assertEqual(report["test_timeout"], ["baseline-sample"])
        self.assertEqual(report["mutation_timeout"], ["baseline-sample"])

    def test_stale_aggregate_log_does_not_contaminate_setting_results(self):
        self.write_predictions([{"id": "sample", "preds": {"full": ["test"]}}])
        self.write_log("sample", "Completed without infrastructure errors.\n", "full")
        self.write_log("sample", INSTALL_FAIL + "\nMutationTimeout")

        report = self.get_report()

        self.assertEqual(report["with_logs"], ["sample"])
        self.assertEqual(report["install_fail"], [])
        self.assertEqual(report["mutation_timeout"], [])

    def test_duplicate_prediction_rows_do_not_duplicate_diagnostics(self):
        row = {"id": "sample", "preds": {"full": ["test"]}}
        self.write_predictions([row, row])
        self.write_log("sample", "MutationTimeout\n", "full")

        report = self.get_report()

        self.assertEqual(report["generated"], ["sample"])
        self.assertEqual(report["with_logs"], ["sample"])
        self.assertEqual(report["mutation_timeout"], ["sample"])


if __name__ == "__main__":
    unittest.main()
