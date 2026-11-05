"""Extract individual Barclays 2024 showings from an existing discovery manifest."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
import platform
from importlib.metadata import version
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

OUTPUT_DIR = INTERIM_DATA_DIR / "events" / "barclays_2024"
CATEGORY_MAPPING_PATH = Path(__file__).with_name("event_categories.json")
SAMPLE_SLUGS = ["brooklyn-nets-vs-new-york-knicks-20240123",
                "new-york-liberty-vs-indiana-fever-20240518", "kehlani-2024",
                "disney-on-ice-2024", "a10-mens-basketball-championship-2024",
                "masters-of-ceremony-2024"]
SAMPLE_URLS = ["https://www.barclayscenter.com/events/detail/" + slug for slug in SAMPLE_SLUGS]
FIELDS = ["performance_id", "showing_id", "venue", "event_name", "event_date", "scheduled_start_time",
          "end_time", "timezone", "category_type", "category_basis", "category_evidence", "event_status",
          "status_evidence", "source_url", "retrieved_at_utc", "raw_path", "raw_sha256",
          "calendar_dates", "calendar_date_match", "date_basis", "start_time_raw", "start_time_status",
          "event_name_missing", "event_date_missing", "start_time_missing",
          "end_time_missing", "category_missing", "status_missing"]
ISSUE_FIELDS = ["severity", "source_url", "showing_id", "calendar_date", "issue", "detail"]


def load_category_mapping(path=CATEGORY_MAPPING_PATH):
    """Load reviewed URL entries; malformed mappings must not silently affect output."""
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Category mapping must be a URL-keyed object")
    for url, entry in mapping.items():
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.netloc != "www.barclayscenter.com"
                or not parts.path.startswith("/events/detail/") or parts.query or parts.fragment):
            raise ValueError(f"Invalid category mapping URL: {url}")
        if not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("category_type", "evidence_location", "evidence")):
            raise ValueError(f"Missing category or evidence in mapping: {url}")
    return mapping


def classify_category(root, name, description):
    """Conservative rules using event-specific text; no artist lookup or tour heuristic."""
    explicit = root.select_one('[itemprop="eventType"], .event-category')
    if explicit:
        value = explicit.get("content") or text(explicit)
        if value:
            return value, "explicit_page_field", value
    if re.search(r"^(?:Brooklyn Nets|New York Liberty)\s+vs\.?\s", name, re.I):
        return "basketball", "game_title_rule", name
    # Only the opening event paragraph, not FAQs or unrelated recommendations.
    paragraph = root.select_one(".descriptionBody p")
    opening = text(paragraph) if paragraph else description
    rules = [(r"\bbasketball\b", "basketball"),
             (r"\bcomedy\b", "comedy"),
             (r"\bconcerts?\b", "concert"),
             (r"\bart exhibit\b", "art_exhibition"),
             (r"\bgymnastics\b", "gymnastics")]
    for pattern, category in rules:
        for evidence in (name, opening):
            if re.search(pattern, evidence, re.I):
                return category, "explicit_event_text_rule", evidence
    return "", "", ""


def text(node):
    return node.get_text(" ", strip=True) if node else ""


def parse_page(html, source_url, calendar_dates, reviewed_categories=None):
    """Extract only explicit showing entries, preserving unknowns and discrepancies."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".event-details")
    if root is None:
        raise ValueError("Missing event-details container")
    name = text(root.select_one("h1.summary"))
    description = text(root.select_one(".descriptionBody"))
    definite_description = " ".join(
        sentence for sentence in re.split(r"(?<=[.!?])\s+", description)
        if not re.search(r"\b(?:if|unless|in case|may|might|could)\b", sentence, re.I))
    # A title is stronger evidence than historical/conditional wording in a description.
    status, status_evidence = "unknown", ""
    for pattern, value in [(r"\bcancel[le]+d\b", "cancelled"),
                           (r"\bpostponed\b", "postponed"),
                           (r"\brescheduled\b", "rescheduled")]:
        if re.search(pattern, name, re.I):
            status, status_evidence = value, name
            break
    if status == "unknown":
        for pattern, value in [(r"\b(?:has|have|is|are)\s+(?:been\s+)?cancel[le]+d\b", "cancelled"),
                               (r"\b(?:has|have|is|are)\s+(?:been\s+)?postponed\b", "postponed"),
                               (r"\b(?:has|have|is|are)\s+(?:been\s+)?rescheduled\b", "rescheduled")]:
            if re.search(pattern, definite_description, re.I):
                status, status_evidence = value, definite_description
                break
    category, basis, category_evidence = classify_category(root, name, description)
    if not category:
        mapping = load_category_mapping() if reviewed_categories is None else reviewed_categories
        reviewed = mapping.get(source_url)
        if reviewed:
            category = reviewed["category_type"]
            basis = "reviewed_url_mapping"
            category_evidence = reviewed["evidence_location"] + ": " + reviewed["evidence"]

    years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(
        text(n) for n in root.select(".details .m-date__year"))))
    rows, issues, seen = [], [], set()
    def issue(kind, detail, showing="", date="", severity="warning"):
        issues.append(dict(severity=severity, source_url=source_url, showing_id=showing,
                           calendar_date=date, issue=kind, detail=detail))
    showings = root.select('.showings li[id^="showing_"]')
    session_numbers = set(re.findall(r"\bSession\s+(\d+)\s*\|", description, re.I))
    if session_numbers and len(session_numbers) != len({s.get("id") for s in showings}):
        issue("session_count_discrepancy",
              f"Description lists {len(session_numbers)} numbered sessions; "
              f"showings list has {len({s.get('id') for s in showings})}. No additional sessions inferred.")
    if not name:
        issue("missing_event_name", "No main event heading")
    # A prose notice on a multi-show page may refer to only one performance.
    # Keep the evidence for review rather than applying it to every showing.
    if status_evidence and status_evidence != name and len(showings) > 1:
        issue("ambiguous_status_scope", status_evidence)
        status = "unknown"
    for showing in showings:
        sid = showing.get("id")
        if sid in seen:
            continue
        seen.add(sid)
        month = text(showing.select_one(".m-date__month"))
        day = text(showing.select_one(".m-date__day"))
        year = text(showing.select_one(".m-date__year")).strip(" ,")
        if not year and len(years) == 1:
            year = next(iter(years))
        date, date_basis = "", ""
        try:
            if year:
                date = datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date().isoformat()
                date_basis = "page_year_and_showing_date"
            else:
                md = datetime.strptime(f"{month} {day} 2000", "%B %d %Y")
                candidates = [d for d in calendar_dates if d[5:] == md.strftime("%m-%d")]
                if len(candidates) != 1:
                    raise ValueError("Cannot resolve showing year")
                date, date_basis = candidates[0], "manifest_year_and_showing_date"
        except ValueError as exc:
            issue("invalid_showing_date", str(exc), sid)
        time_raw = text(showing.select_one(".time"))
        start, time_status = "", "missing"
        if time_raw:
            normalized = re.sub(r"\s+", "", time_raw).upper()
            if normalized in {"TBA", "TBD", "TBC", "TOBEANNOUNCED", "TOBEDETERMINED"}:
                time_status = "explicitly_unknown"
                issue("explicitly_unknown_start_time", time_raw, sid, date, severity="info")
            else:
                try:
                    start = datetime.strptime(normalized, "%I:%M%p").strftime("%H:%M:%S")
                    time_status = "known"
                except ValueError:
                    time_status = "unparsed"
                    issue("unparsed_start_time", time_raw, sid, date)
        else:
            issue("missing_start_time", "No showing start time", sid)
        match = bool(date and date in calendar_dates)
        if date and not match:
            issue("date_not_in_calendar", date, sid)
        if date and not date.startswith("2024-"):
            issue("outside_target_year", date, sid)
        row = dict(performance_id=hashlib.sha256(f"{source_url}#{sid}".encode()).hexdigest()[:20],
                   showing_id=sid,
                   venue="barclays", event_name=name, event_date=date,
                   scheduled_start_time=start, end_time="", timezone="America/New_York",
                   category_type=category, category_basis=basis, category_evidence=category_evidence,
                   event_status=status,
                   status_evidence=status_evidence, source_url=source_url,
                   calendar_dates=json.dumps(sorted(calendar_dates)), calendar_date_match=match,
                   date_basis=date_basis, start_time_raw=time_raw, start_time_status=time_status,
                   event_name_missing=not bool(name), event_date_missing=not bool(date),
                   start_time_missing=not bool(start), end_time_missing=True,
                   category_missing=not bool(category), status_missing=status == "unknown")
        rows.append(row)
    if not rows:
        raise ValueError("No explicit showing entries found; performance count is unknown")
    for date in sorted(set(calendar_dates) - {r["event_date"] for r in rows}):
        issue("calendar_date_without_showing", "No extracted performance for discovery date", date=date)
    return rows, issues


def fetch_page(url, raw_dir, timeout, retries):
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    attempts = []
    for attempt in range(1, retries + 2):
        record = dict(source_url=url, attempt=attempt,
                      retrieved_at_utc=datetime.now(timezone.utc).isoformat())
        body, error = None, None
        try:
            request = Request(url, headers={"User-Agent": "BarclaysEventExtraction/1.0",
                                            "Accept-Encoding": "identity"})
            with urlopen(request, timeout=timeout) as response:
                record.update(http_status=response.status, final_url=response.url)
                body = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                record["charset"] = charset
        except HTTPError as exc:
            body, error = exc.read(), exc
            record.update(http_status=exc.code, final_url=exc.url)
        except (URLError, TimeoutError, OSError) as exc:
            error = exc
        if body is not None:
            path = raw_dir / f"{key}_attempt_{attempt}.body"
            path.write_bytes(body)
            record.update(raw_path=str(path), raw_sha256=hashlib.sha256(body).hexdigest())
        if error is None:
            try:
                decoded = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
                html = decoded.decode(charset)
                html_path = raw_dir / f"{key}_attempt_{attempt}.html"
                html_path.write_text(html, encoding="utf-8")
                record["html_path"] = str(html_path)
            except (OSError, UnicodeError, LookupError, EOFError) as exc:
                error = exc
        if error is None:
            attempts.append(record)
            return html, record, attempts
        record["error"] = f"{type(error).__name__}: {error}"
        attempts.append(record)
        if attempt > retries or isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
            return None, record, attempts
        time.sleep(min(2 ** (attempt - 1), 8))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_records(rows, entries):
    """Summarize duplicate and calendar checks without removing original records."""
    duplicate_counts = {}
    for keys in [("performance_id",), ("source_url", "showing_id"),
                 ("event_name", "event_date", "scheduled_start_time")]:
        counts = Counter(tuple(r[k] for k in keys) for r in rows)
        duplicate_counts["+".join(keys)] = sum(n - 1 for n in counts.values() if n > 1)
    dates = {}
    for row in rows:
        dates.setdefault(row["source_url"], set()).add(row["event_date"])
    return dict(duplicate_counts=duplicate_counts,
                pages_with_records=len(dates),
                pages_without_records=sorted(set(entries) - set(dates)),
                date_discrepancies=[dict(source_url=url,
                                        calendar_only_dates=sorted(expected - dates.get(url, set())),
                                        extracted_only_dates=sorted(dates.get(url, set()) - expected))
                                    for url, expected in entries.items()
                                    if expected != dates.get(url, set())])


def run(manifest, urls=None, timeout=30, retries=2, replay=None):
    reviewed_categories = load_category_mapping()
    mapping_bytes = CATEGORY_MAPPING_PATH.read_bytes()
    entries = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            url = entry["event_url"]
            parts = urlsplit(url)
            if parts.scheme != "https" or parts.netloc != "www.barclayscenter.com" or not parts.path.startswith("/events/detail/"):
                raise ValueError(f"Invalid manifest URL: {url}")
            dates = json.loads(entry["calendar_dates"])
            if not isinstance(dates, list) or not dates:
                raise ValueError(f"Missing calendar dates: {url}")
            for date in dates:
                if datetime.strptime(date, "%Y-%m-%d").year != 2024:
                    raise ValueError(f"Unexpected calendar date: {date}")
            entries.setdefault(url, set()).update(dates)
    if urls:
        if set(urls) - entries.keys():
            raise ValueError("Requested sample URL is absent from manifest")
        entries = {u: entries[u] for u in sorted(set(urls))}
    if not entries:
        raise ValueError("Empty manifest")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = RAW_DATA_DIR / "events" / "barclays_2024" / "details" / run_id
    raw_dir.mkdir(parents=True)
    (raw_dir / "event_categories.json").write_bytes(mapping_bytes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = {}
    if replay:
        previous_summary = json.loads((replay / "summary.json").read_text())
        if previous_summary["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("Replay manifest differs from the saved run")
        cached = {p["source_url"]: p for p in previous_summary["pages"]}
    rows, issues, pages = [], [], []
    failed = 0
    for index, (url, dates) in enumerate(sorted(entries.items()), 1):
        meta, attempts = {}, []
        try:
            if replay:
                page = cached[url]
                meta = page["response"]
                saved_path = Path(meta["raw_path"])
                if not saved_path.exists():
                    saved_path = replay / saved_path.name
                body = saved_path.read_bytes()
                if hashlib.sha256(body).hexdigest() != meta["raw_sha256"]:
                    raise ValueError("Saved raw response hash mismatch")
                if meta.get("error"):
                    raise ValueError(meta["error"])
                html = (gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body).decode(meta.get("charset", "utf-8"))
                attempts = page["attempts"]
            else:
                html, meta, attempts = fetch_page(url, raw_dir, timeout, retries)
            page_record = dict(source_url=url, response=meta, attempts=attempts)
            if html is None:
                raise ValueError(meta["error"])
            extracted, page_issues = parse_page(html, url, dates, reviewed_categories)
            for row in extracted:
                row.update({k: meta[k] for k in ("retrieved_at_utc", "raw_path", "raw_sha256")})
            rows.extend(extracted)
            issues.extend(page_issues)
            page_record["performances"] = len(extracted)
            print(f"[{index}/{len(entries)}] {url.rsplit('/', 1)[-1]}: {len(extracted)} showings", flush=True)
        except (ValueError, OSError, KeyError, LookupError, EOFError) as exc:
            failed += 1
            page_record = dict(source_url=url, error=str(exc),
                               response=meta, attempts=attempts)
            issues.append(dict(severity="error", source_url=url, showing_id="", calendar_date="",
                               issue="page_failure", detail=str(exc)))
            print(f"[{index}/{len(entries)}] FAILED {url}: {exc}", flush=True)
        pages.append(page_record)
        if not replay:
            time.sleep(0.1)
    scope = "sample" if urls else "all"
    output = OUTPUT_DIR / f"events_{scope}.csv"
    issues_path = OUTPUT_DIR / f"extraction_issues_{scope}.csv"
    rows.sort(key=lambda r: (r["event_date"], r["scheduled_start_time"], r["performance_id"]))
    audit = audit_records(rows, entries)
    if any(audit["duplicate_counts"].values()):
        issues.append(dict(severity="warning", source_url="", showing_id="", calendar_date="",
                           issue="duplicate_performances", detail=json.dumps(audit["duplicate_counts"])))
    write_csv(output, rows, FIELDS)
    write_csv(issues_path, issues, ISSUE_FIELDS)
    write_csv(raw_dir / "events.csv", rows, FIELDS)
    write_csv(raw_dir / "issues.csv", issues, ISSUE_FIELDS)
    summary = dict(run_id=run_id, manifest=str(manifest),
                   python_version=platform.python_version(), beautifulsoup_version=version("beautifulsoup4"),
                   parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   category_mapping_sha256=hashlib.sha256(mapping_bytes).hexdigest(),
                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   replay_from=str(replay) if replay else None, pages_requested=len(entries),
                   failed_pages=failed, performances=len(rows),
                   statuses=dict(Counter(r["event_status"] for r in rows)),
                   categories=dict(Counter(r["category_type"] or "unknown" for r in rows)),
                   start_time_statuses=dict(Counter(r["start_time_status"] for r in rows)),
                   issue_severities=dict(Counter(i["severity"] for i in issues)),
                   data_quality=audit,
                   missing_fields={k: sum(bool(r[k]) for r in rows) for k in FIELDS if k.endswith("_missing")},
                   issues=dict(Counter(i["issue"] for i in issues)), pages=pages,
                   output=str(output), issues_output=str(issues_path))
    if urls and set(urls) == set(SAMPLE_URLS):
        by_url = {url: [r for r in rows if r["source_url"] == url] for url in SAMPLE_URLS}
        checks = {"disney_eight_showings": len(by_url[SAMPLE_URLS[3]]) == 8,
                  "a10_seven_sessions": len(by_url[SAMPLE_URLS[4]]) == 7,
                  "masters_cancelled": len(by_url[SAMPLE_URLS[5]]) == 1 and
                  by_url[SAMPLE_URLS[5]][0]["event_status"] == "cancelled",
                  "six_pages_parsed": failed == 0 and all(by_url.values()),
                  "dates_match_manifest": all(r["calendar_date_match"] for r in rows)}
        summary["sample_validation"] = checks
    (raw_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"extraction_summary_{scope}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"extraction_audit_{scope}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pages"}, indent=2))
    print(f"Raw run: {raw_dir}")
    # 1 = script/page failure, 2 = review warning, 0 = success (may include known unknowns).
    if failed or not all(summary.get("sample_validation", {}).values()):
        return 1
    return 2 if any(i["severity"] == "warning" for i in issues) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "discovery_all.csv")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--urls", nargs="+", help="Sample URLs, all of which must be in the manifest")
    selection.add_argument("--sample", action="store_true", help="Validate the six representative pages")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--replay", type=Path, help="Reparse a previous raw run without network requests")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive; retries must be nonnegative")
    return run(args.manifest, SAMPLE_URLS if args.sample else args.urls,
               args.timeout, args.retries, args.replay)


if __name__ == "__main__":
    sys.exit(main())
