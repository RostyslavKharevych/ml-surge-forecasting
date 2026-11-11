"""Concert attendance from tour box scores (Billboard Boxscore figures), basis `tour_boxscore`.

Two published copies of the box scores are read, most trusted first: Wikipedia tour articles'
per-city tables, then Touring Data's per-tour posts for nights Wikipedia lacks. A concert matches
a box-score row on venue name and a date range; a run reported as one figure over several nights
is split evenly across the nights it reports.
"""

import argparse
import json
import logging
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

from src.ingestion.box_scores import fill_blanks
from src.config import ANALYSIS_YEARS, INTERIM_DATA_DIR, RAW_DATA_DIR

log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
TOURING_DATA_API = "https://touringdata.org/wp-json/wp/v2/posts"
BOXSCORE_VENUES = {"barclays": "Barclays Center"}
# Wikimedia asks bots for a descriptive agent and to back off on 429.
USER_AGENT = "SurgeRegressor/1.0 (academic research; github.com/RostyslavKharevych/ml-surge-forecasting)"
RAW_DIR = RAW_DATA_DIR / "tour_boxscores"
ATTENDANCE_PATH = INTERIM_DATA_DIR / "tour_attendance.csv"
ATTENDANCE_COLUMNS = ["event_id", "attendance", "source"]


def _expand(table):
    """A table's rows as lists of (text, rowspan) with rowspans and colspans filled in."""
    grid, carry = [], {}  # carry: column -> [rows left, cell]
    for tr in table.find_all("tr"):
        row, cells, col = [], tr.find_all(["th", "td"], recursive=False), 0
        while cells or carry.get(col, [0])[0] > 0:
            if carry.get(col, [0])[0] > 0:
                carry[col][0] -= 1
                row.append(carry[col][1])
                col += 1
                continue
            cell = cells.pop(0)
            for sup in cell.find_all("sup"):
                sup.decompose()
            span = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            entry = (cell.get_text(" ", strip=True), span)
            for _ in range(int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)):
                if span > 1:
                    carry[col] = [span - 1, entry]
                row.append(entry)
                col += 1
        grid.append(row)
    return grid


def _number(text):
    match = re.match(r"\d[\d,]*", text)
    return int(match.group().replace(",", "")) if match else None


def parse_wikipedia(html, venue_name):
    """(first night, last night, attendance per night) at a venue from every table with Date and
    Attendance columns. Attendance is the first figure of 'sold / available'; a run's shared
    cell spans one row per night."""
    rows = []
    for table in BeautifulSoup(html, "html.parser").find_all("table"):
        att_col = None
        for row in _expand(table):
            texts = [text for text, _ in row]
            dates = [i for i, text in enumerate(texts) if text.startswith("Date")]
            attendances = [i for i, text in enumerate(texts) if text.startswith("Attend")]
            if dates and attendances:  # a header; tables often restart one per year as "Date (2024)"
                date_col, att_col = dates[0], attendances[0]
                year = re.search(r"\d{4}", texts[date_col])
                continue
            if att_col is None or len(row) <= att_col or not any(venue_name in text for text in texts):
                continue
            date_text = texts[date_col]
            if not re.search(r"\d{4}", date_text):
                if not year:  # without a year pandas would guess the current one
                    continue
                date_text += f" {year.group()}"
            date = pd.to_datetime(date_text, errors="coerce")
            attendance, span = _number(row[att_col][0]), row[att_col][1]
            if pd.notna(date) and attendance:
                rows.append((date.date(), date.date(), round(attendance / span)))
    return rows


def _date_range(text):
    """'June 16-17, 2024' or 'May 31-June 1, 2024' -> (first, last) dates."""
    match = re.fullmatch(r"(\w+) (\d+)(?:\s*[-–]\s*(?:(\w+) )?(\d+))?, (\d{4})", text)
    if not match:
        return None
    month, day, last_month, last_day, year = match.groups()
    return (pd.Timestamp(f"{month} {day} {year}").date(),
            pd.Timestamp(f"{last_month or month} {last_day or day} {year}").date())


def parse_touring_data(html, venue_name):
    """(first night, last night, tickets per show) at a venue from a Touring Data post, whose rows
    read 'dates | artist | venue | city' and 'gross | tickets (pct) | N shows'."""
    rows = []
    for tr in BeautifulSoup(html, "html.parser").find_all("tr"):
        cells = [cell.get_text("\n", strip=True).split("\n") for cell in tr.find_all("td")]
        if len(cells) != 2 or len(cells[0]) < 3 or cells[0][2] != venue_name or len(cells[1]) < 2:
            continue
        nights, tickets = _date_range(cells[0][0]), _number(cells[1][1])
        shows = _number(cells[1][2]) if len(cells[1]) > 2 else 1
        if nights and tickets and shows:
            rows.append((*nights, round(tickets / shows)))
    return rows


def _fetch(url, path, parse=json.loads):
    # ponytail: a raw file is the cache; rerun after a failure to resume.
    for wait in (5, 30, 120, None):
        if path.exists():
            break
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
                path.write_bytes(response.read())
            time.sleep(1)
        except HTTPError as error:
            if error.code != 429 or wait is None:
                raise
            time.sleep(wait)
    return parse(path.read_text(encoding="utf-8"))


def _slug(text):
    return re.sub(r"\W+", "_", text).strip("_")[:120]


def _wiki(params, path):
    return _fetch(f"{WIKI_API}?{urlencode({**params, 'format': 'json', 'formatversion': 2})}", path)


def wikipedia_tours(artist, venue_name, year):
    """(article, html) for articles naming the artist that mention the venue and year."""
    found = _wiki({"action": "query", "list": "search", "srlimit": 3,
                   "srsearch": f'{artist} tour "{venue_name}" {year}'},
                  RAW_DIR / f"search_{_slug(artist)}_{year}.json")
    for hit in found["query"]["search"]:
        page = _wiki({"action": "parse", "page": hit["title"], "prop": "text", "redirects": 1},
                     RAW_DIR / f"page_{_slug(hit['title'])}.json")
        yield f"wikipedia:{hit['title']}", page.get("parse", {}).get("text", "")


def touring_data_tours(artist):
    """(post url, html) for Touring Data posts matching the artist."""
    posts = _fetch(f"{TOURING_DATA_API}?{urlencode({'search': artist, 'per_page': 5, '_fields': 'link,content'})}",
                   RAW_DIR / f"touringdata_{_slug(artist)}.json")
    for post in posts:
        yield post["link"], post["content"]["rendered"]


def match_concerts(concerts):
    """event_id, attendance, source for each concert night found in a box score. Touring Data
    is only asked for nights Wikipedia lacks."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    found = []
    for (venue, artist, year), group in concerts.groupby(
            ["venue", "event_name", concerts.event_start.dt.year]):
        venue_name = BOXSCORE_VENUES[venue]
        unmatched = dict(zip(group.event_start.dt.date, group.event_id))
        sources = [(wikipedia_tours(artist, venue_name, year), parse_wikipedia),
                   (touring_data_tours(artist), parse_touring_data)]
        for pages, parse in sources:
            for source, html in pages:
                for first, last, attendance in parse(html, venue_name):
                    for date in [date for date in unmatched if first <= date <= last]:
                        found.append((unmatched.pop(date), attendance, source))
                if not unmatched:
                    break
            if not unmatched:
                break
    return pd.DataFrame(found, columns=ATTENDANCE_COLUMNS)


def held_concerts(events):
    return events[(events.event_type == "concert") & (events.event_status == "held")
                  & events.event_start.notna()]


def apply_tour_attendance(events, found):
    """Fill blank concert attendance from matched box scores; a stale event_id raises."""
    stale = set(found.event_id) - set(events.event_id)
    if stale:
        raise ValueError(f"Tour attendance for unknown events: {sorted(stale)}")
    events = events.copy()
    values = found.set_index("event_id").attendance.reindex(events.event_id).set_axis(events.index)
    fill_blanks(events, "attendance", "attendance_basis", values.astype("Int64"), "tour_boxscore")
    return events


def read_tour_attendance(path=ATTENDANCE_PATH):
    return pd.read_csv(path, dtype={"event_id": str, "source": str}).astype({"attendance": "Int64"})


def main():
    from src.processing.events import load_events  # events reads this module's output

    parser = argparse.ArgumentParser(description=f"Match concerts to tour box scores into {ATTENDANCE_PATH}")
    parser.add_argument("--years", type=int, nargs="+", default=ANALYSIS_YEARS)
    years = parser.parse_args().years
    logging.basicConfig(format="%(message)s")
    concerts = held_concerts(load_events(tour_attendance_path=None, calibrate=False))
    concerts = concerts[concerts.event_start.dt.year.isin(years)]
    found = match_concerts(concerts)
    found.to_csv(ATTENDANCE_PATH, index=False)
    hit = concerts.event_start.dt.year.to_frame("year").assign(hit=concerts.event_id.isin(found.event_id))
    print(hit.groupby("year").hit.agg(["sum", "count", "mean"]).to_string())
    print(concerts.drop(columns="attendance").merge(found, on="event_id", how="left")[
        ["event_id", "event_name", "attendance", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
