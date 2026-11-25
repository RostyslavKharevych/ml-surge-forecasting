## Selecting an event year

Both event scripts accept `--year` (default `2024`). The selected year controls
calendar requests, manifest validation, showing-year checks, and all raw/interim
paths. Existing 2024 commands continue to work. Data is isolated under
`data/raw/events/barclays_<year>/` and `data/interim/events/barclays_<year>/`.
Extraction defaults to that year's `discovery_all.csv`; an explicit `--manifest`
must contain only calendar dates from the selected year. Summaries record the year.

Before collecting the full 2025 year, discover January only:

```bash
python3 -m src.discover_events --year 2025 --months 1
```

Then extract only the first three URLs from that generated manifest (no guessed
URLs). This produces `events_sample.csv`, `extraction_issues_sample.csv`, and the
sample summary/audit in `data/interim/events/barclays_2025/`:

```bash
python3 - <<'PY'
import csv
import subprocess
import sys

manifest = 'data/interim/events/barclays_2025/discovery_months_01.csv'
with open(manifest, newline='', encoding='utf-8') as stream:
    urls = [row['event_url'] for row in csv.DictReader(stream)][:3]
if not urls:
    raise SystemExit('No URLs discovered; review the discovery summary first.')
result = subprocess.run([sys.executable, '-m', 'src.ingest_events', '--year', '2025',
                         '--manifest', manifest, '--urls', *urls])
raise SystemExit(result.returncode)
PY
```

Inspect those outputs and raw pages before expanding the sample to different
event types. The built-in `--sample` remains a 2024-only validation set and is
rejected with `--year 2025`; use `--urls` for other years. No 2025 live-page
structure or historical completeness is established by the offline tests.

After the small 2025 sample is validated, full-year commands are:

```bash
python3 -m src.discover_events --year 2025
python3 -m src.ingest_events --year 2025
```

For offline replay, use the same `--year`, manifest, and URL selection as the saved
run, plus `--replay PATH_TO_RAW_RUN`. Existing 2024 snapshots without an explicit
summary year remain supported through manifest/hash validation.

## Project Structure

```text
├── data/
│   ├── raw/            # Untouched raw data files (ignored by Git)
│   ├── interim/        # Intermediary aggregated tables
│   └── processed/      # Cleaned feature tables ready for modeling
├── notebooks/          # Exploratory analysis & presentation figures
├── src/
│   ├── ingestion/     # TLC downloads, Barclays crawler, Weather & Box Scores APIs
│   ├── processing/    # TLC telemetry cleaning, duration calibration, event merges
│   ├── features/      # Spatial weighting, clean-week baselines, training matrix
│   └── models/        # Baselines, LightGBM/XGBoost regressors, evaluation
├── tests/              # Sanity checks and unit tests
└── requirements.txt    # Environment dependencies
```

## Barclays Center 2024 calendar discovery

From the repository root (Python 3.10+; this component uses only the standard library):

```bash
python3 -m src.discover_events
```

Small live sample and offline parser tests:

```bash
python3 -m src.discover_events --months 1 --timeout 30 --retries 2
python3 -m unittest tests.test_discover_events -v
```

The script requests `/events/calendar/2024/{month}?v=2` for all twelve months by
default. It discovers links only; it does not fetch detail pages or create demand
features. Each canonical event URL has one CSV row with a JSON array of all its
associated ISO calendar dates. These dates are calendar associations, not a count
of performances: detail pages may contain several shows on the same day.

Outputs (already excluded by the existing data ignore rules):

- `data/raw/events/barclays_2024/<run_id>/`: original response bodies for each
  attempt and `summary.json` with request URLs, UTC retrieval times, hashes,
  HTTP statuses, monthly counts, and errors. Gzip bodies remain compressed.
- `data/interim/events/barclays_2024/discovery_all.csv`: deduplicated full-year
  manifest. Sample runs use a separate `discovery_months_01.csv` style filename.

Each run refetches sources and retains its raw snapshot; its scoped CSV is replaced.
Failed months produce a partial manifest and exit status 1; inspect `summary.json`
before using it. Successful requests do not guarantee a complete historical venue
schedule. Cancelled events may be included. Empty months are reported with zero
counts. Timeouts default to 30 seconds, with two retries and exponential backoff;
non-transient HTTP errors are not retried. No credentials are required.

## Barclays Center 2024 event extraction

This stage reads the existing discovery manifest and fetches event detail pages.
It does not discover URLs again. Install dependencies, then validate the six
representative pages (Nets, Liberty, Kehlani, Disney, A-10, Masters of Ceremony):

```bash
python3 -m pip install -r requirements-events.txt
python3 -m src.ingest_events --sample
python3 -m unittest discover -v
```

The sample summary checks eight Disney showings, seven A-10 sessions, cancellation
of Masters of Ceremony, successful parsing of all six pages, and calendar date
agreement. After reviewing the sample, the full manifest can be processed with:

```bash
python3 -m src.ingest_events
```

Use `--manifest PATH` for another discovery manifest or `--urls URL [URL ...]`
to select specific URLs already present in that manifest. `--timeout` defaults to
30 seconds and `--retries` to 2. HTTP 408/429 and selected server errors are retried;
other HTTP errors are reported immediately. Requests are sequential with a short
delay. Parser failures are retained for review rather than silently skipped.

Local outputs under `data/interim/events/barclays_2024/`:

- `events_sample.csv` / `events_all.csv`: one row per explicit showing ID, with
  name, date, local scheduled start, category basis, status evidence, source URL,
  retrieval timestamp, raw response hash/path, and missing-value flags.
- `extraction_issues_sample.csv` / `extraction_issues_all.csv`: request failures,
  unparsed dates/times, ambiguous status notices, and calendar discrepancies.
- `extraction_summary_sample.json` / `extraction_summary_all.json`: page counts,
  performance counts, statuses, missing fields, issues, and request histories.

Each run preserves original response bytes (`.body`), decoded UTF-8 HTML (`.html`),
and immutable CSV/summary snapshots under
`data/raw/events/barclays_2024/details/<run_id>/`. Generated data is ignored by Git.
To reparse a saved run without accessing the network, use the same selection:

```bash
python3 -m src.ingest_events --sample --replay data/raw/events/barclays_2024/details/<run_id>
```

Replay verifies raw hashes and retains the original retrieval timestamps. Every
requested URL is represented in the summary, including failures. Failed pages
do not create invented performances; consult the issues CSV alongside records.
Exit status 0 means extraction succeeded (explicitly unknown source times may
remain). Status 1 means an actual request/parsing failure or failed sample
validation; status 2 means a data-quality warning requires review. Issues have
`info`, `warning`, or `error` severity. Partial output remains available.

The parser uses the main event container and explicit showing list, excluding
unrelated upcoming recommendations and sidebar headline times. A listed session
may contain several sports games; the script does not invent individual game
start times. Dates are cross-checked in both directions against the manifest.
The page year is preferred; an unambiguous manifest year is used only when the
showing omits it, with `date_basis` recording this choice.

Unknown starts and end times remain blank. `start_time_status` distinguishes
`known`, `explicitly_unknown` (TBA/TBD/TBC), `missing`, and `unparsed`. The original
time text is preserved; explicitly unknown times are informational, not parser
failures. Categories come from explicit fields, the Nets/Liberty game-title rule,
or literal event-specific terms (`basketball`, `comedy`, `concert`, `art exhibit`,
`gymnastics`) in the main title or opening description paragraph. Each derived
category records its rule and evidence. Artist names, decorative icons, and
the word "tour" alone do not establish a category.

When these rules cannot determine a category, the parser consults
`src/event_categories.json`, a manually reviewed mapping keyed by exact canonical
event URL. It contains 44 approved 2024 entries: the original six plus 30 additional
mappings using existing labels and eight supported event types using new labels.
The new labels are `community_event`, `music_dance_event`, `listening_party`,
`boxing`, `sports_draft`, `bull_riding`, and `wrestling` (two WWE pages).
Evidence comes from saved official event descriptions
or event-specific metadata. Existing rule classifications always take precedence;
unmapped events remain blank. Mapped rows use `category_basis=reviewed_url_mapping`
and record the source field and supporting text in `category_evidence`.

Each entry must include `category_type`, `evidence_location`, `evidence`, and an
integer `reviewed_year`. The 44 original entries are reviewed for 2024 and apply
only when `--year 2024` is selected. An additional 40 entries are reviewed for
2025: they classify 60 performances, including four wine/sake experiences under
`food_drink_event`. These reviews use saved official descriptions or event-specific
metadata. Remaining 2025 categories stay blank (`unknown` in summary counts).
A URL without a year in its name is not
automatically safe for reuse. The general category rules still apply in 2025;
otherwise categories remain blank until supported, year-specific review occurs.
Review the source content before adding entries; an artist name or "tour" alone
is insufficient. The mapping is validated before extraction, copied into each
raw run, and hashed in the summary. Replay uses the current reviewed mapping so
saved responses can be reclassified; the mapping hash identifies the version.
The first six mappings reduced unknown categories from 96 to 76. The additional
38 mappings reduce unknowns to 32, covering 154 of the 186 performances. All
23 uncertain URLs across 22 event names remain unmapped (Zach Bryan has two URLs).
Mappings are URL-specific: reviewing Aventura's December page or Usher's December
page does not classify their other pages. Descriptive evidence takes priority
over conflicting metadata in reviewed entries: the boxing event is not a concert,
and Billie Eilish's album listening party is distinct from a live concert.

The 2025 snapshot retains 166 performances from 141 pages. Reviewed mappings
increase category coverage from 79 to 139 performances; 27 performances across
22 URLs remain uncertain. The R&B Forever Tour notice identifies its original
November 9, 2024 schedule and ticket validity on the rescheduled April 26, 2025
date. Status detection preserves both dates in `status_evidence` while retaining
only the explicit April 26 showing. Old schedule mentions or conditional ticket
notices alone do not establish a rescheduling. The resulting statuses are three
cancelled, two rescheduled, and 161 unknown; these are not occurrence confirmations.

No confirmed-completed status is inferred from historical listings.
Definite cancellation/postponement/rescheduling notices are preserved, while
conditional notices and ambiguous multi-show prose do not become definite
performance statuses. Date/status exceptions require review before modeling.

The summary includes duplicate and bidirectional calendar-date checks, also saved
as `extraction_audit_sample.json` / `extraction_audit_all.json`. It records Python
and BeautifulSoup versions plus script and input hashes. Replay verifies the
manifest and response hashes. Pinned event-only dependencies avoid installing
the unrelated modeling packages; discovery itself uses only Python's standard
library. Tested with Python 3.14; Python 3.10+ is required.

Known limitations from the 2024 snapshot: 186 explicit showings are retained from
148 pages. Three start times are TBA. The NCAA March Madness description lists
three sessions while its explicit showing list contains only two; this produces
a review warning, with no inferred third showing. Unknown categories and actual
completion statuses remain unresolved, and end times are not inferred. Website
retention does not prove a complete venue calendar. Future live runs may reflect
site edits; saved raw responses support reproducible offline parsing. Review
the issues CSV and summary alongside the event records.