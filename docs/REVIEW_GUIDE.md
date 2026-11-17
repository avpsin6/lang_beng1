# Review guide

What was built while you were away, in the order it is easiest to review,
with the judgement calls made in each phase and the open questions.

## Your answers

| Question | Answer | Effect |
|---|---|---|
| Interface | Report + dashboard | Two offline HTML files |
| Priorities | Sleep, heart & fitness, activity & workouts, change over years | All four built |
| Sleep data | Rarely / not sure | Sleep code treats "in bed" and measured sleep separately and copes with gaps |
| Privacy | Never send data anywhere | No network code; pages block network access |

## Phases (one commit each, in `git log`)

| Phase | What to look at | Decisions made for you |
|---|---|---|
| 0–1 Scaffold, generator, parser | `parser.py`, `synthetic.py`, `tests/test_parser.py` | Data lives in `~/HealthLens`, not the repo. Parquet cache. Parser repairs malformed exports instead of failing. |
| 2 Metrics | `metrics/` | Watch > iPhone > apps priority merge. Nights dated by the morning. Missing is not zero. |
| 3 Findings | `analysis/` | Conservative statistics; personal baseline first; cited references; no diagnostic wording (a test enforces this). |
| 4 Report + CLI | `report/`, `templates/report.html.j2`, `cli.py` | Charting library inlined so pages work offline (about 6 MB each). |
| 5 Dashboard | `static/dashboard.js` | A generated HTML file, not a local server: nothing to keep running, no telemetry. |
| 6 Docs and checks | `README.md`, `docs/` | — |

## What was verified

* 45 automated tests pass on Python 3.13 / pandas 3.1 and Python 3.9 / pandas 2.3.
* Both pages rendered in headless Chromium, light and dark, desktop and phone
  width: no script errors, no network requests, no sideways scrolling.
* A 900 MB synthetic export (2.1 million records) ran end to end in 40 s with
  about 840 MB peak memory.

## Not verified

* **No real Apple Health export has been run.** The parser follows Apple's
  documented format and handles old and new variants, but the first run on
  your export is the real test. If it fails, `healthlens info` and the error
  message are the things to send back (not the export itself).
* Totals have not been compared with the Health app on a real device.

## Open questions for you

1. After your first real run: do daily step totals match the Health app for a
   few days you spot-check?
2. Which Watch model is it? `healthlens info` will show it. That decides
   whether ECG or blood-oxygen sections are worth adding.
3. Do you want 24-hour or am/pm times, and metric or imperial, regardless of
   the iPhone setting?
4. Is the wording at the right level, or would you like it shorter or more technical?
5. The repository is public. Do you want it made private, or a licence added?
6. Possible next steps: ECG summaries, route maps (offline, no map tiles),
   a printable one-page summary for a doctor, comparing two exports over time.
