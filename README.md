# ML Surge Forecasting

Tabular time-series regression forecasting engine to predict granular demand surges and net deltas around key event locations.

## Project Structure

```text
├── data/
│   ├── raw/            # Untouched raw data files (ignored by Git)
│   ├── interim/        # Intermediary aggregated tables
│   └── processed/      # Cleaned feature tables ready for modeling
├── notebooks/          # Exploratory analysis & presentation figures
├── src/                # Modular pipeline scripts
│   ├── config.py       # Global paths, constants, and column names
│   ├── ingest.py       # Raw file acquisition
│   ├── clean.py        # Schema enforcement & parsing
│   ├── features.py     # Lags, rolling windows, calendar indicators
│   ├── baseline.py     # Naive median / control comparisons
│   ├── train.py        # Model training (LightGBM, XGBoost, Ridge)
│   └── evaluate.py     # Metrics (MAE, RMSE, R2) & SHAP attribution
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
event URL. It contains 44 approved entries: the original six plus 30 additional
mappings using existing labels and eight supported event types using new labels.
The new labels are `community_event`, `music_dance_event`, `listening_party`,
`boxing`, `sports_draft`, `bull_riding`, and `wrestling` (two WWE pages).
Evidence comes from saved official event descriptions
or event-specific metadata. Existing rule classifications always take precedence;
unmapped events remain blank. Mapped rows use `category_basis=reviewed_url_mapping`
and record the source field and supporting text in `category_evidence`.

Each entry must include `category_type`, `evidence_location`, and `evidence`.
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

