"""Discover Barclays Center 2024 event pages; no event detail requests are made."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

BASE_URL = "https://www.barclayscenter.com"
YEAR = 2024


class EventLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        parts = urlsplit(urljoin(BASE_URL, href))
        if (parts.scheme in {"http", "https"}
                and parts.netloc.lower() in {"www.barclayscenter.com", "barclayscenter.com"}
                and parts.path.startswith("/events/detail/")
                and parts.path.removeprefix("/events/detail/").strip("/")):
            self.urls.add(urlunsplit(("https", "www.barclayscenter.com",
                                     parts.path.rstrip("/"), "", "")))


def parse_calendar(payload, month):
    """Return URL -> ISO calendar dates; reject unexpected schemas and dates."""
    if not isinstance(payload, dict):
        raise ValueError("Calendar response must be a date-to-HTML object")
    events = {}
    for date_text, fragment in payload.items():
        date = datetime.strptime(date_text, "%m-%d-%Y").date()
        if date.year != YEAR or date.month != month:
            raise ValueError(f"Unexpected date for month {month}: {date_text}")
        if not isinstance(fragment, str):
            raise ValueError(f"Expected HTML string for {date_text}")
        parser = EventLinks()
        parser.feed(fragment)
        if fragment.strip() and not parser.urls:
            raise ValueError(f"Nonempty calendar entry has no event links: {date_text}")
        for url in parser.urls:
            events.setdefault(url, set()).add(date.isoformat())
    return events


def fetch_calendar(url, timeout, retries, raw_dir, month):
    """Preserve response bytes, including HTTP error bodies, for each attempt."""
    attempts = []
    for attempt in range(1, retries + 2):
        record = {"attempt": attempt, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        body = None
        try:
            request = Request(url, headers={"User-Agent": "BarclaysEventDiscovery/1.0",
                                            "Accept": "application/json",
                                            "Accept-Encoding": "identity"})
            try:
                with urlopen(request, timeout=timeout) as response:
                    record.update(status=response.status, final_url=response.url)
                    body = response.read()
                    encoding = response.headers.get("Content-Encoding", "")
                    record["content_type"] = response.headers.get("Content-Type")
            except HTTPError as exc:
                body = exc.read()
                record.update(status=exc.code, final_url=exc.url)
                raise
            finally:
                if body is not None:
                    raw_path = raw_dir / f"calendar_{YEAR}-{month:02d}_attempt_{attempt}.body"
                    raw_path.write_bytes(body)
                    record.update(raw_path=str(raw_path), sha256=hashlib.sha256(body).hexdigest())
            decoded = gzip.decompress(body) if encoding == "gzip" or body[:2] == b"\x1f\x8b" else body
            payload = json.loads(decoded)
            events = parse_calendar(payload, month)
            attempts.append(record)
            return events, len(payload), attempts, None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)
            retryable = not isinstance(exc, HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
            if attempt > retries or not retryable:
                return {}, 0, attempts, record["error"]
            time.sleep(min(2 ** (attempt - 1), 8))


def run(months, timeout=30, retries=2):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = RAW_DATA_DIR / "events" / f"barclays_{YEAR}" / run_id
    raw_dir.mkdir(parents=True, exist_ok=False)
    output_dir = INTERIM_DATA_DIR / "events" / f"barclays_{YEAR}"
    output_dir.mkdir(parents=True, exist_ok=True)
    events, provenance, summaries = {}, {}, []
    for month in sorted(set(months)):
        url = f"{BASE_URL}/events/calendar/{YEAR}/{month}?v=2"
        found, date_count, attempts, error = fetch_calendar(url, timeout, retries, raw_dir, month)
        summaries.append({"month": month, "request_url": url, "unique_urls": len(found),
                          "calendar_dates": date_count, "error": error, "attempts": attempts})
        for event_url, dates in found.items():
            events.setdefault(event_url, set()).update(dates)
            provenance.setdefault(event_url, set()).add(url)
        print(f"{YEAR}-{month:02d}: {len(found)} URLs, {date_count} dates"
              + (f" FAILED: {error}" if error else ""), flush=True)

    # Separate names prevent a sample run from overwriting the full-year manifest.
    scope = "all" if len(set(months)) == 12 else "months_" + "_".join(f"{m:02d}" for m in sorted(set(months)))
    manifest = output_dir / f"discovery_{scope}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["venue", "event_url", "calendar_dates",
                                                    "source_calendar_urls", "run_id"])
        writer.writeheader()
        for event_url in sorted(events):
            writer.writerow({"venue": "barclays", "event_url": event_url,
                             "calendar_dates": json.dumps(sorted(events[event_url])),
                             "source_calendar_urls": json.dumps(sorted(provenance[event_url])),
                             "run_id": run_id})
    failures = sum(item["error"] is not None for item in summaries)
    summary = {"year": YEAR, "run_id": run_id, "months": summaries,
               "unique_urls": len(events), "failed_months": failures,
               "complete_year_requests": len(set(months)) == 12 and failures == 0,
               "manifest": str(manifest)}
    (raw_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Total: {len(events)} unique URLs; {failures} failed months\nManifest: {manifest}\nRaw responses and summary: {raw_dir}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", type=int, choices=range(1, 13), default=list(range(1, 13)))
    parser.add_argument("--timeout", type=float, default=30, help="Socket timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the initial attempt")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive and retries must be nonnegative")
    return run(args.months, args.timeout, args.retries)


if __name__ == "__main__":
    sys.exit(main())
