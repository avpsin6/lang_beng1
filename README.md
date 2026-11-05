# HealthLens

Private, offline analysis of your Apple Health / Apple Watch export: activity,
heart & fitness, sleep, workouts and how everything changed over the years.

> **Work in progress.** See `docs/REVIEW_GUIDE.md` for the build phases and the
> decisions made along the way.

## Privacy first

* Runs entirely on your computer. No network requests, no telemetry, no AI services.
* Your data and reports live in `~/HealthLens/` — **outside this repository**.
  This repository is public; `.gitignore` additionally blocks exports, caches
  and reports from ever being committed.
