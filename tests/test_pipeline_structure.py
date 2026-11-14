from pathlib import Path

import pandas as pd

from src.config import (
    CITI_FIELD_CORE_ZONE,
    CITI_FIELD_PICKUP_ZONES,
    CITI_FIELD_SPILLOVER_ZONES,
    get_citi_field_tlc_dir,
)
from src.ingestion import citi_field_ingest
from src.processing.clean_tlc import aggregate_pickups_15min, clean_pickups
from src.processing.merge_events import (
    add_event_end_times,
    label_pickup_intervals,
    prepare_game_event_windows,
)


def test_shared_zone_config_and_raw_directory():
    assert CITI_FIELD_CORE_ZONE == 93
    assert CITI_FIELD_SPILLOVER_ZONES == {253}
    assert CITI_FIELD_PICKUP_ZONES == {93, 253}
    assert get_citi_field_tlc_dir() == Path(__file__).parents[1] / "data/raw/tlc/citi_field"


def test_clean_and_aggregate_tlc_pickups_for_core_and_spillover_zones():
    trips = pd.DataFrame(
        {
            "pickup_datetime": [
                "2024-07-01 19:02:00",
                "2024-07-01 19:09:00",
                "2024-07-01 19:12:00",
                "not-a-time",
            ],
            "PULocationID": [93, 253, 7, 253],
        }
    )

    clean = clean_pickups(trips)
    combined, by_zone = aggregate_pickups_15min(clean)

    assert len(clean) == 2
    assert combined.to_dict("records") == [
        {"time_bin": pd.Timestamp("2024-07-01 19:00:00"), "pickup_count": 2}
    ]
    assert by_zone["PULocationID"].tolist() == [93, 253]
    assert by_zone["pickup_count"].tolist() == [1, 1]


def test_game_window_labeling_excludes_concerts_and_respects_boundaries():
    events = pd.DataFrame(
        [
            {
                "event_id": "game-1",
                "event_type": "game",
                "event_date": "2024-07-01",
                "opponent": "Team A",
                "start_time_local": "2024-07-01 19:00:00-0400",
                "duration_minutes": 120,
                "end_time_local": None,
            },
            {
                "event_id": "concert-1",
                "event_type": "concert",
                "event_date": "2024-07-01",
                "opponent": None,
                "start_time_local": "2024-07-01 18:00:00-0400",
                "duration_minutes": None,
                "end_time_local": None,
            },
        ]
    )
    events = add_event_end_times(events)
    games = prepare_game_event_windows(events)
    pickups = pd.DataFrame(
        {
            "time_bin": pd.to_datetime(
                [
                    "2024-07-01 18:15",
                    "2024-07-01 18:30",
                    "2024-07-01 19:00",
                    "2024-07-01 21:00",
                    "2024-07-01 22:29",
                    "2024-07-01 22:30",
                ]
            ),
            "pickup_count": [0, 1, 2, 3, 4, 0],
        }
    )

    labeled = label_pickup_intervals(pickups, games)

    assert games.event_id.tolist() == ["game-1"]
    assert labeled.event_phase.tolist() == [
        "no_event",
        "before_game",
        "during_game",
        "after_game",
        "after_game",
        "no_event",
    ]
    assert labeled.loc[1, "opponent"] == "Team A"


def test_event_ingestion_combines_game_rows_and_concert_rows(monkeypatch):
    game_row = {
        "event_id": "METS_20240701_1",
        "event_type": "game",
        "event_name": "Mets vs. Team A",
        "event_date": "2024-07-01",
        "venue": "Citi Field",
        "venue_zone_id": 93,
        "opponent": "Team A",
        "start_time_local": "2024-07-01 19:00:00-0400",
        "attendance": None,
        "duration_minutes": 120,
        "mlb_game_id": 1,
        "end_time_local": "2024-07-01 21:00:00-0400",
    }
    monkeypatch.setattr(
        citi_field_ingest,
        "fetch_games",
        lambda year: pd.DataFrame([game_row]),
    )

    events = citi_field_ingest.fetch_events(2024)

    assert len(events) == 5
    assert events.event_type.value_counts().to_dict() == {"game": 1, "concert": 4}
    assert events.event_id.is_unique
    assert events.event_date.is_monotonic_increasing
