import unittest
from pathlib import Path

from src.ingest_events import load_category_mapping, parse_page
from tests.test_ingest_events import page


class Categories2025Tests(unittest.TestCase):
    def test_reviewed_2025_entries_and_year_isolation(self):
        mapping = load_category_mapping()
        reviewed = {u: e for u, e in mapping.items() if e["reviewed_year"] == 2025}
        self.assertEqual(len(reviewed), 40)
        self.assertEqual(sum(e["reviewed_year"] == 2024 for e in mapping.values()), 44)
        self.assertEqual(sum(e["category_type"] == "food_drink_event" for e in reviewed.values()), 4)
        for url, entry in reviewed.items():
            with self.subTest(url=url):
                rows, _ = parse_page(page().replace(", 2024", ", 2025"), url,
                                     ["2025-01-18"], year=2025)
                self.assertEqual(rows[0]["category_type"], entry["category_type"])
                self.assertEqual(rows[0]["category_basis"], "reviewed_url_mapping")
                self.assertIn(entry["evidence"], rows[0]["category_evidence"])
                rows, _ = parse_page(page(), url, ["2024-01-18"])
                self.assertEqual(rows[0]["category_type"], "")

    def test_2024_mappings_do_not_leak_into_2025(self):
        for url, entry in load_category_mapping().items():
            if entry["reviewed_year"] != 2024:
                continue
            with self.subTest(url=url):
                rows, _ = parse_page(page().replace(", 2024", ", 2025"), url,
                                     ["2025-01-18"], year=2025)
                self.assertEqual(rows[0]["category_type"], "")

    def test_uncertain_2025_urls_remain_unknown(self):
        slugs = """anuel-aa-2025 christian-nodal david-tao druski-2025 gem
            jonas-brothers-2025 kaytranada-x-justice katt-williams-2025 konstantinos-argiros
            lil-baby-2025 linkin-park-2025 planet-brooklyn playboi-carti-2025 rb-forever-tour
            rain vybz-kartel reggae-fest-massive-vybz-kartel-labor-day-weekend renee-rapp
            rod-wave-12-2025 ryan-castro tame-impala-2025 mgk""".split()
        self.assertEqual(len(slugs), 22)
        mapping = load_category_mapping()
        for slug in slugs:
            with self.subTest(slug=slug):
                url = "https://www.barclayscenter.com/events/detail/" + slug
                self.assertNotIn(url, mapping)
                rows, _ = parse_page(page().replace(", 2024", ", 2025"), url,
                                     ["2025-01-18"], year=2025)
                self.assertEqual(rows[0]["category_type"], "")

    def test_rb_forever_real_notice_preserves_both_dates(self):
        fixture = Path(__file__).parent / "fixtures/barclays_2025/rb-forever-tour.html"
        rows, issues = parse_page(fixture.read_text(),
                                 "https://www.barclayscenter.com/events/detail/rb-forever-tour",
                                 ["2025-04-26"], year=2025)
        self.assertEqual(issues, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_date"], "2025-04-26")
        self.assertEqual(rows[0]["event_status"], "rescheduled")
        self.assertIn("November 9, 2024", rows[0]["status_evidence"])
        self.assertIn("April 26, 2025", rows[0]["status_evidence"])
        self.assertEqual(rows[0]["category_type"], "")
        self.assertEqual(rows[0]["end_time"], "")

    def test_old_date_or_conditional_ticket_notice_is_not_definite_status(self):
        for description in [
            "The show was originally scheduled on November 9, 2024.",
            "The show was originally scheduled on November 9, 2024. "
            "If moved, tickets will automatically be valid for the rescheduled date.",
        ]:
            with self.subTest(description=description):
                rows, _ = parse_page(page(description=description),
                                     "https://www.barclayscenter.com/events/detail/example", ["2024-01-18"])
                self.assertEqual(rows[0]["event_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
