import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.run_pipeline import PipelineStep, build_steps, main, parse_args, run_pipeline


class RunPipelineTests(unittest.TestCase):
    def test_default_step_order_and_flags(self):
        steps = build_steps()
        self.assertEqual(
            [step.module for step in steps],
            [
                "src.ingestion.tlc",
                "src.ingestion.weather",
                "src.ingestion.citi_field_ingest",
                "src.ingestion.barclays_discovery",
                "src.ingestion.barclays_ingest",
                "src.ingestion.barclays_discovery",
                "src.ingestion.barclays_ingest",
                "src.ingestion.box_scores",
                "src.ingestion.tour_boxscores",
                "src.processing.clean_tlc",
                "src.features.features",
                "src.features.dataset",
            ],
        )
        self.assertEqual(steps[0].args, ("--years", "2024", "2025", "--delete-raw"))
        self.assertEqual(steps[2].args, ("--years", "2024", "2025"))
        self.assertEqual(steps[4].args, ("--year", "2024"))
        self.assertEqual(steps[4].allowed_exit_codes, frozenset({0, 2}))
        self.assertEqual(steps[6].args, ("--year", "2025"))
        self.assertEqual(steps[6].allowed_exit_codes, frozenset({0, 2}))
        self.assertEqual(steps[8].args, ("--years", "2024", "2025"))

    def test_skip_and_keep_raw_flags(self):
        cached = build_steps(skip_tlc=True, skip_scrape=True)
        self.assertEqual(
            [step.module for step in cached],
            [
                "src.processing.clean_tlc",
                "src.features.features",
                "src.features.dataset",
            ],
        )

        keep_raw = build_steps(years=[2024], skip_scrape=True, delete_raw=False)
        self.assertEqual(keep_raw[0].args, ("--years", "2024"))

        ns = parse_args(["--years", "2024", "--skip-tlc", "--skip-scrape", "--keep-raw"])
        self.assertEqual(ns.years, [2024])
        self.assertTrue(ns.skip_tlc)
        self.assertTrue(ns.skip_scrape)
        self.assertTrue(ns.keep_raw)

    def test_run_pipeline_allows_exit_code_2_and_stops_on_error(self):
        calls = []

        def fake_run(cmd, check=False):
            calls.append((cmd, check))
            code = 2 if "barclays_ingest" in cmd else 0
            return subprocess.CompletedProcess(cmd, code)

        steps = [
            PipelineStep("Barclays Scrape", "src.ingestion.barclays_ingest", ("--year", "2024"), frozenset({0, 2})),
            PipelineStep("Dataset", "src.features.dataset"),
        ]
        with patch("scripts.run_pipeline.subprocess.run", side_effect=fake_run):
            run_pipeline(steps, python_exe=sys.executable)

        self.assertEqual(
            calls,
            [
                ([sys.executable, "-m", "src.ingestion.barclays_ingest", "--year", "2024"], False),
                ([sys.executable, "-m", "src.features.dataset"], False),
            ],
        )

        failing_calls = []

        def fake_fail(cmd, check=False):
            failing_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1)

        with patch("scripts.run_pipeline.subprocess.run", side_effect=fake_fail):
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                run_pipeline(steps, python_exe=sys.executable)
        self.assertEqual(ctx.exception.returncode, 1)
        self.assertEqual(len(failing_calls), 1)

    def test_main_runs_cached_pipeline(self):
        invoked = []

        def fake_run(cmd, check=False):
            invoked.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        with patch("scripts.run_pipeline.subprocess.run", side_effect=fake_run):
            rc = main(["--skip-tlc", "--skip-scrape"])

        self.assertEqual(rc, 0)
        self.assertEqual(
            [cmd[2] for cmd in invoked],
            ["src.processing.clean_tlc", "src.features.features", "src.features.dataset"],
        )
