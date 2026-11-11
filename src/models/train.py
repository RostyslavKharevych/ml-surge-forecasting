"""Model training pipeline for Surge Delta prediction (Ridge, LightGBM, XGBoost)."""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train surge forecasting models.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/event_dataset.parquet"),
        help="Path to processed training dataset",
    )
    args = parser.parse_args()

    print(f"Dataset path configured: {args.dataset}")
    # Modeling logic to be implemented


if __name__ == "__main__":
    main()