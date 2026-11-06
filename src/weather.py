"""Venue Weather: recorded hourly weather at each arena from the Open-Meteo archive,
copied onto the 15-minute buckets (each bucket takes its hour's reading).

Times are naive America/New_York, like the rest of the pipeline.
"""

import argparse
import json
from urllib.request import urlopen

import pandas as pd

from src.config import ANALYSIS_YEARS, INTERIM_DATA_DIR, VENUE_CONFIG

ARCHIVE_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               "&start_date={year}-01-01&end_date={year}-12-31"
               "&hourly=precipitation,temperature_2m,wind_speed_10m"
               "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=mm"
               "&timezone=America%2FNew_York")
WEATHER_DIR = INTERIM_DATA_DIR / "weather"
RAIN_MM = 0.1  # more than a trace
QUARTERS = pd.to_timedelta([0, 15, 30, 45], unit="min")


def to_buckets(archive, venue):
    """One row per 15-minute bucket from an Open-Meteo hourly archive response."""
    hourly = archive["hourly"]
    hours = pd.DataFrame({"venue": venue, "hour": pd.to_datetime(hourly["time"]),
                          "precipitation_mm": hourly["precipitation"],
                          "temp_f": hourly["temperature_2m"],
                          "wind_mph": hourly["wind_speed_10m"]})
    buckets = hours.loc[hours.index.repeat(len(QUARTERS))].reset_index(drop=True)
    buckets.insert(1, "timestamp_15m", buckets.pop("hour") + list(QUARTERS) * len(hours))
    buckets["is_raining"] = buckets.precipitation_mm > RAIN_MM
    return buckets


def fetch_archive(venue, year):
    # ponytail: the raw response is the cache, so a rerun makes no requests.
    path = WEATHER_DIR / f"{venue}_{year}.json"
    if not path.exists():
        WEATHER_DIR.mkdir(parents=True, exist_ok=True)
        lat, lon = VENUE_CONFIG[venue]["coordinates"]
        with urlopen(ARCHIVE_URL.format(lat=lat, lon=lon, year=year), timeout=60) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text(encoding="utf-8"))


def load_weather(venues=VENUE_CONFIG, years=ANALYSIS_YEARS):
    """Venue Weather for every venue and year, one row per venue x 15-minute bucket."""
    return pd.concat([to_buckets(fetch_archive(venue, year), venue)
                      for venue in venues for year in years], ignore_index=True)


def main():
    argparse.ArgumentParser(description=f"Fetch Venue Weather into {WEATHER_DIR}").parse_args()
    weather = load_weather()
    print(weather.groupby("venue").agg(buckets=("timestamp_15m", "size"),
                                       rainy_share=("is_raining", "mean"),
                                       temp_f=("temp_f", "mean")))


if __name__ == "__main__":
    main()
