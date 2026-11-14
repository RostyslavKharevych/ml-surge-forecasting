"""Read pickup columns from monthly NYC TLC HVFHV parquet files."""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.config import CITI_FIELD_PICKUP_ZONES, get_citi_field_tlc_dir


def find_trip_files(data_dir: Optional[Path] = None) -> list:
    """Find monthly parquet inputs in the configured Citi Field raw-data folder."""
    source_dir = Path(data_dir) if data_dir is not None else get_citi_field_tlc_dir()
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No TLC parquet files found in {}".format(source_dir))
    return files


def load_pickup_columns(
    files: Iterable[Path], pickup_zones=CITI_FIELD_PICKUP_ZONES
) -> pd.DataFrame:
    """Load pickup time and zone fields, retaining requested pickup zones."""
    chunks = []
    for file_path in files:
        trips = pd.read_parquet(
            file_path, columns=["pickup_datetime", "PULocationID"]
        )
        chunks.append(trips[trips["PULocationID"].isin(pickup_zones)].copy())

    if not chunks:
        return pd.DataFrame(columns=["pickup_datetime", "PULocationID"])
    return pd.concat(chunks, ignore_index=True)
