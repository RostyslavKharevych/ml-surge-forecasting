import json
import unittest
from pathlib import Path

import pandas as pd

from src.box_scores import apply_box_scores, parse_summary
from src.events import from_barclays_scrape

FIXTURES = Path(__file__).parent / "fixtures" / "espn"


def summary(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def events(*rows):
    """Scraped showings as (name, date, start, category) tuples."""
    return from_barclays_scrape(pd.DataFrame([dict(
        venue="barclays", event_name=name, event_date=date, scheduled_start_time=start,
        category_type=category, event_status="unknown") for name, date, start, category in rows]))


def game(game_id, tip, end, attendance):
    return dict(venue="barclays", game_id=game_id, game_name=f"game {game_id}",
                tip=pd.Timestamp(tip), end=pd.Timestamp(end) if end else pd.NaT,
                attendance=attendance)


class ParseSummaryTests(unittest.TestCase):
    def test_wnba_tip_end_and_attendance_in_local_time(self):
        parsed = parse_summary(summary("wnba_401721255.json"))
        self.assertEqual(parsed, dict(venue="barclays", game_id="401721255",
                                      tip=pd.Timestamp("2024-09-22 13:04:32"),
                                      end=pd.Timestamp("2024-09-22 14:58:03"),
                                      attendance=12115))

    def test_implausible_end_is_dropped(self):
        # ESPN once shifted a Nets game's wallclock by a day mid-game.
        game = summary("wnba_401721255.json")
        game["plays"][-1]["wallclock"] = "2024-09-23T18:58:03Z"
        self.assertTrue(pd.isna(parse_summary(game)["end"]))

    def test_zero_attendance_is_missing(self):
        self.assertTrue(pd.isna(parse_summary(summary("ncaam_401638614.json"))["attendance"]))


class MatchTests(unittest.TestCase):
    def test_single_game_sets_box_score_end_and_attendance(self):
        table = apply_box_scores(events(("Nets vs. Heat", "2024-01-15", "19:30:00", "basketball")),
                                 pd.DataFrame([game("1", "2024-01-15 19:41", "2024-01-15 21:58", 17000)]))
        event = table.iloc[0]
        self.assertEqual(event.event_end, pd.Timestamp("2024-01-15 21:58"))
        self.assertEqual((event.end_time_basis, event.attendance, event.attendance_basis),
                         ("box_score", 17000, "box_score"))

    def test_session_ends_at_last_game_and_games_go_to_latest_prior_start(self):
        table = apply_box_scores(
            events(("NCAA", "2024-03-22", "12:15:00", "basketball"),
                   ("NCAA", "2024-03-22", "19:10:00", "basketball")),
            pd.DataFrame([game("1", "2024-03-22 12:16", "2024-03-22 14:48", None),
                          game("2", "2024-03-22 15:23", "2024-03-22 17:22", 17183),
                          game("3", "2024-03-22 19:11", "2024-03-22 21:15", None),
                          game("4", "2024-03-22 21:50", "2024-03-23 00:01", 17487)]))
        self.assertEqual(list(table.event_end), [pd.Timestamp("2024-03-22 17:22"),
                                                 pd.Timestamp("2024-03-23 00:01")])
        self.assertEqual(list(table.attendance), [17183, 17487])

    def test_unmatched_events_are_left_blank_and_logged(self):
        with self.assertLogs("src.box_scores", "WARNING") as logs:
            table = apply_box_scores(
                events(("Harlem Globetrotters", "2024-02-18", "13:00:00", "basketball"),
                       ("Kehlani", "2024-02-18", "20:00:00", "concert")),
                pd.DataFrame([game("9", "2024-03-01 19:40", "2024-03-01 21:50", 17000)]))
        self.assertTrue(table.event_end.isna().all())
        self.assertEqual(set(table.end_time_basis), {""})
        output = "\n".join(logs.output)
        self.assertIn("Harlem Globetrotters", output)
        self.assertNotIn("Kehlani", output)
        self.assertIn("game 9", output)

    def test_game_without_end_leaves_end_blank_but_keeps_attendance(self):
        table = apply_box_scores(events(("Liberty vs. Sky", "2024-06-01", "13:00:00", "basketball")),
                                 pd.DataFrame([game("1", "2024-06-01 13:05", None, 12000)]))
        self.assertTrue(pd.isna(table.iloc[0].event_end))
        self.assertEqual(table.iloc[0].attendance, 12000)

    def test_reviewed_end_is_not_overwritten(self):
        table = events(("Nets vs. Heat", "2024-01-15", "19:30:00", "basketball"))
        table.loc[0, ["event_end", "end_time_basis"]] = [pd.Timestamp("2024-01-15 22:10"), "broadcast_window"]
        table = apply_box_scores(table, pd.DataFrame(
            [game("1", "2024-01-15 19:41", "2024-01-15 21:58", 17000)]))
        self.assertEqual((table.iloc[0].event_end, table.iloc[0].end_time_basis),
                         (pd.Timestamp("2024-01-15 22:10"), "broadcast_window"))
        self.assertEqual(table.iloc[0].attendance_basis, "box_score")


if __name__ == "__main__":
    unittest.main()
