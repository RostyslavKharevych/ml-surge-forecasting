import contextlib
import csv
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from src import discover_events, ingest_events
from tests.test_ingest_events import page


class Response(io.BytesIO):
    def __init__(self, body, url, content_type):
        super().__init__(body)
        self.status = 200
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type


class EventYearTests(unittest.TestCase):
    def test_discovery_selected_year_and_default_validation(self):
        payload = {"01-18-2025": '<a href="/events/detail/example">Example</a>'}
        records = discover_events.parse_calendar(payload, 1, year=2025)
        self.assertEqual(next(iter(records.values())), {"2025-01-18"})
        with self.assertRaises(ValueError):
            discover_events.parse_calendar(payload, 1)
        with self.assertRaises(ValueError):
            discover_events.parse_calendar({"01-18-2024": ""}, 1, year=2025)
        with self.assertRaises(ValueError):
            discover_events.parse_calendar({"02-29-2025": ""}, 2, year=2025)

    def test_extraction_year_check_preserves_cross_year_evidence(self):
        html = page().replace(", 2024", ", 2025")
        rows, issues = ingest_events.parse_page(html, "https://www.barclayscenter.com/events/detail/example",
                                               ["2025-01-18"], year=2025)
        self.assertEqual(issues, [])
        self.assertEqual(rows[0]["event_date"], "2025-01-18")
        _, issues = ingest_events.parse_page(html, "https://www.barclayscenter.com/events/detail/example",
                                            ["2025-01-18"])
        self.assertIn("outside_target_year", {i["issue"] for i in issues})

    def test_year_scoped_mapping_for_reusable_url(self):
        url = "https://www.barclayscenter.com/events/detail/chase-atlantic"
        html = page().replace(", 2024", ", 2025")
        rows, _ = ingest_events.parse_page(html, url, ["2025-01-18"], year=2025)
        self.assertEqual(rows[0]["category_type"], "")
        mapping = {url: dict(category_type="concert", evidence="Reviewed event evidence",
                             evidence_location=".descriptionBody", reviewed_year=2025)}
        rows, _ = ingest_events.parse_page(html, url, ["2025-01-18"], mapping, year=2025)
        self.assertEqual(rows[0]["category_basis"], "reviewed_url_mapping")
        rows, _ = ingest_events.parse_page(page(), url, ["2024-01-18"], mapping)
        self.assertEqual(rows[0]["category_type"], "")

    def test_general_category_rules_remain_available_in_2025(self):
        html = page(name="Brooklyn Nets vs. Example").replace(", 2024", ", 2025")
        rows, _ = ingest_events.parse_page(html, "https://www.barclayscenter.com/events/detail/example",
                                          ["2025-01-18"], year=2025)
        self.assertEqual(rows[0]["category_type"], "basketball")

    def test_reviewed_year_required_and_integer(self):
        url = "https://www.barclayscenter.com/events/detail/example"
        for value in [None, "2024", True, 1800]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mapping.json"
                entry = dict(category_type="concert", evidence="Concert", evidence_location="description")
                if value is not None:
                    entry["reviewed_year"] = value
                path.write_text(json.dumps({url: entry}))
                with self.assertRaises(ValueError):
                    ingest_events.load_category_mapping(path)

    def test_cli_year_and_default(self):
        for module in (discover_events, ingest_events):
            for args, expected in [([], 2024), (["--year", "2025"], 2025)]:
                with self.subTest(module=module.__name__, args=args), \
                     patch("sys.argv", [module.__name__] + args), \
                     patch.object(module, "run", return_value=0) as mocked:
                    self.assertEqual(module.main(), 0)
                    self.assertEqual(mocked.call_args.kwargs["year"], expected)

    def test_sample_is_2024_only(self):
        with patch("sys.argv", ["ingest_events", "--year", "2025", "--sample"]), \
             patch.object(ingest_events, "run") as mocked, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                ingest_events.main()
            self.assertEqual(exc.exception.code, 2)
            mocked.assert_not_called()
        with patch("sys.argv", ["ingest_events", "--sample"]), \
             patch.object(ingest_events, "run", return_value=0) as mocked:
            ingest_events.main()
            self.assertEqual(mocked.call_args.args[1], ingest_events.SAMPLE_URLS)

    def test_selected_year_rejects_other_year_manifest_before_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "discovery.csv"
            manifest.write_text('event_url,calendar_dates\n'
                                'https://www.barclayscenter.com/events/detail/example,"[""2024-01-18""]"\n')
            with patch.object(ingest_events, "fetch_page") as fetch:
                with self.assertRaises(ValueError):
                    ingest_events.run(manifest, year=2025)
                fetch.assert_not_called()

    def test_discovery_and_extraction_output_isolation_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interim, raw = root / "interim", root / "raw"
            with patch.object(discover_events, "INTERIM_DATA_DIR", interim), \
                 patch.object(discover_events, "RAW_DATA_DIR", raw), \
                 patch.object(ingest_events, "INTERIM_DATA_DIR", interim), \
                 patch.object(ingest_events, "OUTPUT_DIR", interim / "events/barclays_2024"), \
                 patch.object(ingest_events, "RAW_DATA_DIR", raw), \
                 patch("src.ingest_events.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                snapshot = {}
                for year in (2024, 2025):
                    def calendar_response(request, timeout):
                        self.assertIn(f"/calendar/{year}/1?", request.full_url)
                        payload = {f"01-18-{year}": '<a href="/events/detail/example">Example</a>'}
                        return Response(json.dumps(payload).encode(), request.full_url, "application/json")
                    with patch.object(discover_events, "urlopen", side_effect=calendar_response):
                        self.assertEqual(discover_events.run([1], year=year), 0)
                    output = interim / "events" / f"barclays_{year}"
                    manifest = output / "discovery_all.csv"
                    manifest.write_bytes((output / "discovery_months_01.csv").read_bytes())
                    html = page().replace(", 2024", f", {year}")
                    with patch.object(ingest_events, "urlopen", side_effect=lambda request, timeout:
                                      Response(html.encode(), request.full_url, "text/html; charset=utf-8")):
                        self.assertEqual(ingest_events.run(year=year), 0)
                    with (output / "events_all.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows[0]["event_date"], f"{year}-01-18")
                    summary = json.loads((output / "extraction_summary_all.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["year"], year)
                    self.assertEqual(Path(summary["manifest"]), manifest)
                    self.assertIn(f"barclays_{year}", rows[0]["raw_path"])
                    self.assertTrue((output / "extraction_audit_all.json").exists())
                    self.assertTrue(list((raw / "events" / f"barclays_{year}").glob(f"*/calendar_{year}-01_attempt_1.body")))
                    if year == 2024:
                        snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
                self.assertTrue(all(p.read_bytes() == content for p, content in snapshot.items()))


if __name__ == "__main__":
    unittest.main()
