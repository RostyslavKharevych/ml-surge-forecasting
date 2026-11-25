import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.processing.events import from_barclays_scrape
from src.ingestion.tour_boxscores import apply_tour_attendance, parse_touring_data, parse_wikipedia

FIXTURES = Path(__file__).parent / "fixtures" / "tour_boxscores"
# A Touring Data row, as in https://touringdata.org/2024/07/10/chris-brown-1111-tour/
TOURING_DATA = """<table><tr><td>June 14, 2024<br>Chris Brown<br>TD Garden<br>Boston, United States</td>
<td>$2,338,078<br>13,775 (100%)<br>1 show</td></tr>
<tr><td>June 16-17, 2024<br>Chris Brown<br>Barclays Center<br>New York, United States</td>
<td>$4,680,305<br>27,018 (100%)<br>2 shows</td></tr>
<tr><td>May 31-June 1, 2024<br>Chris Brown<br>Barclays Center<br>New York, United States</td>
<td>$1<br>20,000 (100%)<br>2 shows</td></tr></table>"""


class ParseTests(unittest.TestCase):
    def test_wikipedia_run_split_per_night_with_year_from_header(self):
        html = (FIXTURES / "wikipedia_usher_past_present_future.html").read_text(encoding="utf-8")
        # Four Barclays nights share one 58,041 cell; two December nights share 26,655.
        self.assertEqual(parse_wikipedia(html, "Barclays Center"), [
            (date(2024, 9, d), date(2024, 9, d), 14510) for d in (6, 7, 9, 10)] + [
            (date(2024, 12, d), date(2024, 12, d), 13328) for d in (5, 6)])

    def test_touring_data_runs_split_by_reported_shows(self):
        self.assertEqual(parse_touring_data(TOURING_DATA, "Barclays Center"), [
            (date(2024, 6, 16), date(2024, 6, 17), 13509),
            (date(2024, 5, 31), date(2024, 6, 1), 10000)])


class ApplyTests(unittest.TestCase):
    def events(self):
        return from_barclays_scrape(pd.DataFrame([dict(
            venue="barclays", event_name="Usher", event_date="2024-09-06", scheduled_start_time="20:00:00",
            category_type="concert", event_status="unknown")]))

    def test_fills_blank_attendance_with_tour_boxscore_basis(self):
        found = pd.DataFrame({"event_id": ["BARCLAYS_20240906_2000"], "attendance": [14510], "source": ["x"]})
        event = apply_tour_attendance(self.events(), found).iloc[0]
        self.assertEqual((event.attendance, event.attendance_basis), (14510, "tour_boxscore"))

    def test_stale_event_id_fails(self):
        found = pd.DataFrame({"event_id": ["BARCLAYS_20240907_2000"], "attendance": [1], "source": ["x"]})
        with self.assertRaisesRegex(ValueError, "unknown events"):
            apply_tour_attendance(self.events(), found)


if __name__ == "__main__":
    unittest.main()
