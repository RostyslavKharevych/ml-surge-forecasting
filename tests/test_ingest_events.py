import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.ingestion.barclays_ingest import parse_page, run, audit_records


def page(name="Example", shows=None, description=""):
    if shows is None:
        shows = [("1", "January", "18", "7:00PM")]
    items = "".join(
        f'<li id="showing_{sid}"><span class="m-date__month">{month}</span>'
        f'<span class="m-date__day">{day}</span><span class="time">{start}</span></li>'
        for sid, month, day, start in shows)
    return (f'<div class="event-details"><h1 class="summary">{name}</h1>'
            '<div class="details"><span class="m-date__year">, 2024</span></div>'
            f'<div class="showings"><ul>{items}</ul></div>'
            f'<div class="descriptionBody">{description}</div></div>'
            '<div><li id="showing_999">Upcoming unrelated event</li></div>')


class ExtractionTests(unittest.TestCase):
    def test_real_page_fixtures(self):
        directory = Path(__file__).parent / "fixtures" / "barclays_2024"
        for case in json.loads((directory / "cases.json").read_text(encoding="utf-8")):
            with self.subTest(page=case["file"]):
                rows, issues = parse_page((directory / case["file"]).read_text(encoding="utf-8"),
                                         case["source_url"], case["calendar_dates"])
                self.assertEqual(len(rows), case["expected_showings"])
                self.assertEqual([i["issue"] for i in issues], case.get("expected_issues", []))
                self.assertTrue(all(r["event_status"] == case["expected_status"] for r in rows))
                self.assertTrue(all(r["event_name"] and r["event_date"] for r in rows))
                self.assertEqual(sum(not r["scheduled_start_time"] for r in rows),
                                 case.get("expected_unknown_times", 0))
                self.assertEqual(len({r["showing_id"] for r in rows}), len(rows))

    def parse(self, html, dates=("2024-01-18",)):
        return parse_page(html, "https://www.barclayscenter.com/events/detail/example", dates)

    def test_multiple_times_and_duplicate_ids(self):
        rows, issues = self.parse(page(shows=[("1", "January", "18", "11:00AM"),
                                              ("2", "January", "18", "7:00PM"),
                                              ("2", "January", "18", "7:00PM")]))
        self.assertEqual([r["scheduled_start_time"] for r in rows], ["11:00:00", "19:00:00"])
        self.assertEqual(issues, [])
        self.assertEqual([r["showing_id"] for r in rows], ["showing_1", "showing_2"])

    def test_missing_time_and_category_are_not_invented(self):
        rows, issues = self.parse(page(shows=[("1", "January", "18", "TBA")]))
        self.assertEqual(rows[0]["scheduled_start_time"], "")
        self.assertTrue(rows[0]["start_time_missing"])
        self.assertEqual(rows[0]["end_time"], "")
        self.assertEqual(rows[0]["category_type"], "")
        self.assertEqual(rows[0]["event_status"], "unknown")
        self.assertEqual(rows[0]["start_time_status"], "explicitly_unknown")
        self.assertEqual(issues[0]["issue"], "explicitly_unknown_start_time")
        self.assertEqual(issues[0]["severity"], "info")

    def test_categories_require_event_specific_evidence(self):
        for name, description, expected in [
            ("Example World Tour", "Example brings the World Tour to Brooklyn.", ""),
            ("A-10 Men's Basketball Championship", "", "basketball"),
            ("Example", "The Example concert has been rescheduled.", "concert"),
            ("Example Comedy Jam", "", "comedy"),
            ("The Basquiat Project", "An art exhibit featuring students' work.", "art_exhibition")]:
            with self.subTest(name=name):
                rows, _ = self.parse(page(name=name, description=description))
                self.assertEqual(rows[0]["category_type"], expected)
                if expected:
                    self.assertTrue(rows[0]["category_evidence"])

    def test_session_count_discrepancy_retains_only_explicit_showings(self):
        rows, issues = self.parse(page(description="Session 1 | Games. Session 2 | Games."))
        self.assertEqual(len(rows), 1)
        self.assertEqual(issues[0]["issue"], "session_count_discrepancy")
        self.assertEqual(issues[0]["severity"], "warning")

    def test_malformed_time_is_warning(self):
        rows, issues = self.parse(page(shows=[("1", "January", "18", "25:00PM")]))
        self.assertEqual(rows[0]["start_time_status"], "unparsed")
        self.assertEqual(issues[0]["severity"], "warning")

    def test_duplicate_audit_retains_records(self):
        rows, _ = self.parse(page())
        rows += rows.copy()
        audit = audit_records(rows, {rows[0]["source_url"]: {"2024-01-18"}})
        self.assertEqual(audit["duplicate_counts"]["performance_id"], 1)
        self.assertEqual(audit["date_discrepancies"], [])
        self.assertEqual(len(rows), 2)

    def test_info_and_warning_exit_codes(self):
        for start, expected in [("TBA", 0), ("25:00PM", 2)]:
            with self.subTest(start=start), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "manifest.csv"
                url = "https://www.barclayscenter.com/events/detail/example"
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["event_url", "calendar_dates"])
                    writer.writeheader()
                    writer.writerow(dict(event_url=url, calendar_dates='["2024-01-18"]'))
                meta = dict(retrieved_at_utc="2026-11-04T12:00:00+00:00", raw_path="saved.body", raw_sha256="hash")
                with patch("src.ingestion.barclays_ingest.OUTPUT_DIR", root / "output"), \
                     patch("src.ingestion.barclays_ingest.RAW_DATA_DIR", root / "raw"), \
                     patch("src.ingestion.barclays_ingest.fetch_page", return_value=(page(shows=[("1", "January", "18", start)]), meta, [meta])), \
                     patch("src.ingestion.barclays_ingest.time.sleep"), patch("builtins.print"):
                    self.assertEqual(run(manifest), expected)

    def test_cancellation_wins_over_rescheduling(self):
        rows, _ = self.parse(page(name="Cancelled: Example", description="This has been rescheduled."))
        self.assertEqual(rows[0]["event_status"], "cancelled")

    def test_postponement_and_game_category(self):
        rows, _ = self.parse(page(name="New York Liberty vs. Indiana Fever", description="This event has been postponed."))
        self.assertEqual(rows[0]["event_status"], "postponed")
        self.assertEqual(rows[0]["category_type"], "basketball")

    def test_crosscheck_both_directions(self):
        rows, issues = self.parse(page(), ("2024-01-19",))
        self.assertFalse(rows[0]["calendar_date_match"])
        self.assertEqual({i["issue"] for i in issues},
                         {"date_not_in_calendar", "calendar_date_without_showing"})

    def test_missing_showings_is_failure(self):
        with self.assertRaises(ValueError):
            self.parse(page(shows=[]))

    def test_conditional_cancellation_is_not_status(self):
        rows, _ = self.parse(page(description="If the event is cancelled, contact your seller."))
        # Conditional notices must not be mistaken for a definite cancellation.
        self.assertEqual(rows[0]["event_status"], "unknown")

    def test_partial_status_notice_is_flagged_for_review(self):
        html = page(shows=[("1", "January", "18", "7:00PM"),
                           ("2", "January", "19", "7:00PM")],
                    description="The January 18 show has been postponed.")
        rows, issues = self.parse(html, ("2024-01-18", "2024-01-19"))
        self.assertTrue(all(r["event_status"] == "unknown" for r in rows))
        self.assertIn("ambiguous_status_scope", {i["issue"] for i in issues})

    def test_leap_day_without_page_year(self):
        html = page(shows=[("1", "February", "29", "7:00PM")])
        html = html.replace('<span class="m-date__year">, 2024</span>', "")
        rows, issues = self.parse(html, ("2024-02-29",))
        self.assertEqual(rows[0]["event_date"], "2024-02-29")
        self.assertEqual(rows[0]["date_basis"], "manifest_year_and_showing_date")
        self.assertEqual(issues, [])

    def test_failed_page_is_in_summary_and_issues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            url = "https://www.barclayscenter.com/events/detail/example"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["event_url", "calendar_dates"])
                writer.writeheader()
                writer.writerow(dict(event_url=url, calendar_dates='["2024-01-18"]'))
            meta = dict(error="HTTPError: 404", source_url=url)
            with patch("src.ingestion.barclays_ingest.OUTPUT_DIR", root / "output"), \
                 patch("src.ingestion.barclays_ingest.RAW_DATA_DIR", root / "raw"), \
                 patch("src.ingestion.barclays_ingest.fetch_page", return_value=(None, meta, [meta])), \
                 patch("src.ingestion.barclays_ingest.time.sleep"), patch("builtins.print"):
                result = run(manifest)
            self.assertEqual(result, 1)
            summary = json.loads((root / "output/extraction_summary_all.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_pages"], 1)
            with (root / "output/extraction_issues_all.csv").open() as stream:
                issues = list(csv.DictReader(stream))
            self.assertEqual(issues[0]["source_url"], url)
            self.assertEqual(issues[0]["issue"], "page_failure")


if __name__ == "__main__":
    unittest.main()
