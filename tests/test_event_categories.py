import json
import tempfile
import unittest
from pathlib import Path

from src.ingest_events import load_category_mapping, parse_page
from tests.test_ingest_events import page


class ReviewedCategoryTests(unittest.TestCase):
    def test_six_approved_url_entries(self):
        mapping = load_category_mapping()
        expected = {
            "disney-on-ice-2024": "family_show",
            "ringling-bros-and-barnum-bailey-2024": "circus",
            "harlem-globetrotters-2024": "basketball",
            "kehlani-2024": "concert",
            "iron-maiden-2024": "concert",
            "ed-sheeran-2024": "concert",
        }
        self.assertEqual({url.rsplit("/", 1)[-1]: entry["category_type"]
                          for url, entry in mapping.items()}, expected)
        for url, entry in mapping.items():
            with self.subTest(url=url):
                mapped, issues = parse_page(page(), url, ["2024-01-18"])
                self.assertEqual(issues, [])
                self.assertEqual(mapped[0]["category_type"], entry["category_type"])
                self.assertEqual(mapped[0]["category_basis"], "reviewed_url_mapping")
                self.assertIn(entry["evidence"], mapped[0]["category_evidence"])
                self.assertIn(entry["evidence_location"], mapped[0]["category_evidence"])

    def test_existing_rules_take_precedence(self):
        url = "https://www.barclayscenter.com/events/detail/kehlani-2024"
        for html in [page(name="Example Comedy Show"),
                     page().replace('<h1 class="summary">',
                                    '<meta itemprop="eventType" content="explicit_category"><h1 class="summary">')]:
            with self.subTest(html=html):
                baseline, _ = parse_page(html, url, ["2024-01-18"], reviewed_categories={})
                mapped, _ = parse_page(html, url, ["2024-01-18"])
                self.assertEqual(mapped, baseline)
                self.assertNotEqual(mapped[0]["category_basis"], "reviewed_url_mapping")

    def test_fallback_changes_only_category_fields(self):
        url = "https://www.barclayscenter.com/events/detail/disney-on-ice-2024"
        html = page(name="Cancelled: Example", shows=[("1", "January", "18", "TBA")])
        baseline, baseline_issues = parse_page(html, url, ["2024-01-18"], reviewed_categories={})
        mapped, issues = parse_page(html, url, ["2024-01-18"])
        allowed = {"category_type", "category_basis", "category_evidence", "category_missing"}
        self.assertEqual({k: v for k, v in baseline[0].items() if k not in allowed},
                         {k: v for k, v in mapped[0].items() if k not in allowed})
        self.assertEqual(issues, baseline_issues)

    def test_unmapped_event_remains_unknown(self):
        rows, _ = parse_page(page(name="Example World Tour"),
                             "https://www.barclayscenter.com/events/detail/unreviewed", ["2024-01-18"])
        self.assertEqual(rows[0]["category_type"], "")
        self.assertTrue(rows[0]["category_missing"])

    def test_invalid_or_unsupported_mapping_fails_clearly(self):
        url = "https://www.barclayscenter.com/events/detail/example"
        for mapping in [[], {url: {"category_type": "concert"}},
                        {"https://example.com/event": {"category_type": "concert",
                                                      "evidence_location": "description",
                                                      "evidence": "Concert"}}]:
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mapping.json"
                path.write_text(json.dumps(mapping))
                with self.assertRaises(ValueError):
                    load_category_mapping(path)


if __name__ == "__main__":
    unittest.main()
