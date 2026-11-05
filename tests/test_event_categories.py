import json
import tempfile
import unittest
from pathlib import Path

from src.ingest_events import load_category_mapping, parse_page
from tests.test_ingest_events import page


class ReviewedCategoryTests(unittest.TestCase):
    def test_approved_url_entries(self):
        mapping = {url: entry for url, entry in load_category_mapping().items()
                   if entry["reviewed_year"] == 2024}
        expected = {
            "disney-on-ice-2024": "family_show",
            "ringling-bros-and-barnum-bailey-2024": "circus",
            "harlem-globetrotters-2024": "basketball",
            "kehlani-2024": "concert",
            "iron-maiden-2024": "concert",
            "ed-sheeran-2024": "concert",
            "brooklyn-raga-massive-festival": "concert",
            "hot-wheels-monster-trucks-2024": "family_show",
            "nick-cannon-presents-wild-n-out-live-2024": "comedy",
            "aapi-night-market-2024": "community_event",
            "afrobeats-vs-amapiano-takeover-2024": "music_dance_event",
            "billie-eilish-2024": "listening_party",
            "devin-haney-vs-ryan-garcia": "boxing",
            "2024-nba-draft": "sports_draft",
            "professional-bull-riders-2024": "bull_riding",
            "wwe-friday-night-smackdown-2024": "wrestling",
            "wwe-monday-night-raw-202404": "wrestling",
        }
        concert_slugs = """ana-gabriel-2024 aventura-2024 barak-2024 chase-atlantic
            chayanne childish-gambino-2024 christian-nodal-2024 don-toliver-2024
            funk-flex-winter-fest-2024 grupo-frontera-2024 jhene-aiko-2024
            jonathan-lee-2024 kirk-franklin-2024 kygo-2024 latto-2024
            linkin-park-2024 marc-anthony-2024 maxwell-2024 missy-elliott
            nicki-minaj-pink-friday-2-2024 reggae-fest-massive-2024 russ-2024
            sabrina-carpenter-2024 strength-of-a-woman-festival-summit-2024
            thirty-seconds-to-mars-2024 twenty-one-pilots-2024 usher-122024""".split()
        expected.update({slug: "concert" for slug in concert_slugs})
        self.assertEqual(len(expected), 44)
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

    def test_23_uncertain_urls_across_22_names_remain_unmapped(self):
        slugs = """aventura bad-bunny-2024 buju-banton-2024 chris-brown-2024
            don-omar-2024 future-metro-boomin-2024 janet-jackson-2024 joker-xue-2024
            justin-timberlake-2024 kacey-musgraves-2024 ladies-night-rb-super-jam-2024
            luis-miguel-2024 mariah-carey maverick-city-music-2024 peso-pluma
            rb-music-experience-2024 sexyy-red-2024 subtronics-2024
            the-love-hard-tour-2024 the-new-york-salsa-festival-2024 usher
            zach-bryan-2024 zach-bryan-12-2024""".split()
        # The review listed 22 names; Zach Bryan has two distinct source pages.
        self.assertEqual(len(slugs), 23)
        mapping = load_category_mapping()
        for slug in slugs:
            with self.subTest(slug=slug):
                url = "https://www.barclayscenter.com/events/detail/" + slug
                self.assertNotIn(url, mapping)
                rows, _ = parse_page(page(), url, ["2024-01-18"])
                self.assertEqual(rows[0]["category_type"], "")

    def test_description_evidence_beats_misleading_concert_metadata(self):
        for slug, category in [("devin-haney-vs-ryan-garcia", "boxing"),
                               ("billie-eilish-2024", "listening_party"),
                               ("nick-cannon-presents-wild-n-out-live-2024", "comedy")]:
            with self.subTest(slug=slug):
                html = '<meta name="keywords" content="Concert">' + page()
                rows, _ = parse_page(html, "https://www.barclayscenter.com/events/detail/" + slug,
                                     ["2024-01-18"])
                self.assertEqual(rows[0]["category_type"], category)
                self.assertIn(".descriptionBody", rows[0]["category_evidence"])

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
