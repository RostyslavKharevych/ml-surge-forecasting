"""Single-command Step 1 data pipeline runner for SurgeEgress."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import sys
import time
from typing import Sequence

from src.config import ANALYSIS_YEARS


@dataclass(frozen=True)
class PipelineStep:
    """Single module invocation inside the Step 1 pipeline."""

    label: str
    module: str
    args: tuple[str, ...] = ()
    allowed_exit_codes: frozenset[int] = frozenset({0})


def build_steps(
    years: Sequence[int] = ANALYSIS_YEARS,
    skip_tlc: bool = False,
    skip_scrape: bool = False,
    delete_raw: bool = True,
) -> list[PipelineStep]:
    """Build the ordered list of Step 1 pipeline modules and CLI flags."""
    year_args = tuple(str(y) for y in years)
    steps: list[PipelineStep] = []

    if not skip_tlc:
        tlc_args = ("--years", *year_args, *(["--delete-raw"] if delete_raw else []))
        steps.append(PipelineStep("TLC HVFHV Ingestion", "src.ingestion.tlc", tlc_args))

    if not skip_scrape:
        steps.extend([
            PipelineStep("Open-Meteo Weather Ingestion", "src.ingestion.weather"),
            PipelineStep(
                "Citi Field Event Ingestion",
                "src.ingestion.citi_field_ingest",
                ("--years", *year_args),
            ),
        ])
        for year in years:
            y_arg = ("--year", str(year))
            steps.append(
                PipelineStep(
                    f"Barclays Event Discovery ({year})",
                    "src.ingestion.barclays_discovery",
                    y_arg,
                )
            )
            steps.append(
                PipelineStep(
                    f"Barclays Event Scrape ({year})",
                    "src.ingestion.barclays_ingest",
                    y_arg,
                    allowed_exit_codes=frozenset({0, 2}),
                )
            )
        steps.extend([
            PipelineStep("ESPN Basketball Box Scores", "src.ingestion.box_scores"),
            PipelineStep(
                "Concert Tour Box Scores",
                "src.ingestion.tour_boxscores",
                ("--years", *year_args),
            ),
        ])

    steps.extend([
        PipelineStep("TLC Trip Cleaning", "src.processing.clean_tlc"),
        PipelineStep("15-Min Demand Feature Matrix", "src.features.features"),
        PipelineStep("Event & Model Dataset Assembly", "src.features.dataset"),
    ])
    return steps


def run_pipeline(steps: Sequence[PipelineStep], python_exe: str = sys.executable) -> float:
    """Execute pipeline steps sequentially, printing stage banners and timings."""
    total = len(steps)
    pipeline_start = time.perf_counter()

    for idx, step in enumerate(steps, start=1):
        cmd = [python_exe, "-m", step.module, *step.args]
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{idx}/{total}] {step.label}: {' '.join(cmd)}", flush=True)
        print(f"{'=' * 72}", flush=True)

        step_start = time.perf_counter()
        result = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - step_start

        if result.returncode not in step.allowed_exit_codes:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        print(
            f"[{idx}/{total}] Completed {step.label} in {elapsed:.2f}s (exit={result.returncode})",
            flush=True,
        )

    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\nPipeline completed ({total} steps) in {total_elapsed:.2f}s.", flush=True)
    return total_elapsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Step 1 pipeline runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(ANALYSIS_YEARS))
    parser.add_argument(
        "--skip-tlc",
        action="store_true",
        help="Skip downloading/filtering monthly TLC HVFHV parquets.",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip weather, Citi Field, Barclays, ESPN, and tour box-score ingestion.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Retain raw citywide TLC parquet downloads instead of passing --delete-raw.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    steps = build_steps(
        years=args.years,
        skip_tlc=args.skip_tlc,
        skip_scrape=args.skip_scrape,
        delete_raw=not args.keep_raw,
    )
    run_pipeline(steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
