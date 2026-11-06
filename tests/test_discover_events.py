import unittest

from src.discover_events import parse_calendar


class CalendarDiscoveryTests(unittest.TestCase):
    def test_canonical_links_preserve_multiple_dates(self):
        records = parse_calendar({
            "01-18-2024": '<a href="/events/detail/show?type=premium">Show</a>'
                          '<a href="https://example.com/events/detail/other">Other</a>',
            "01-19-2024": '<a href="http://barclayscenter.com/events/detail/show/#cal">Show</a>'
                          '<a href="/events/detail/show">Duplicate</a>',
        }, 1)
        self.assertEqual(records, {
            "https://www.barclayscenter.com/events/detail/show": {"2024-01-18", "2024-01-19"}
        })

    def test_reject_unexpected_payloads(self):
        for payload in ([], {"02-01-2024": "HTML"}, {"01-01-2024": None},
                        {"01-01-2024": "<p>Changed markup</p>"}):
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                parse_calendar(payload, 1)

    def test_empty_month_is_valid(self):
        self.assertEqual(parse_calendar({}, 1), {})


if __name__ == "__main__":
    unittest.main()
