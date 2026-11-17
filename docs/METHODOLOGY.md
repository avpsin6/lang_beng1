# How HealthLens makes its numbers

## Reading the export

* The export XML is streamed straight out of `export.zip`, so memory stays
  bounded for multi-GB files. The DTD is dropped, characters that are illegal
  in XML are removed, and if parsing still fails the file is re-read in a
  repair mode that replaces invalid text.
* Each sample keeps its UTC time and the UTC offset written in the export.
  A "day" is the local calendar day under that offset. Apple writes the
  offsets of the phone that made the export, so days from long trips abroad
  can be shifted; sleep and workouts use their own stored time zone when present.
* The day the export was made is partial and is left out.

## Counting things once

Your iPhone and Watch both count steps, distance and flights. The export
contains both. HealthLens merges them the way the Health app does:

1. Sources are ranked Watch, then iPhone, then other apps, then manual entries.
2. A lower-ranked sample only counts for the part of its time span that no
   higher-ranked sample covers (pro-rated by time).
3. A sample longer than 3 hours (an app writing a daily total) adds only what
   exceeds the higher-ranked total in its span.

Totals can differ slightly from the Health app, which lets you reorder
sources per data type. Exact duplicate samples are dropped.

## Missing is not zero

Days with no data stay blank. Exercise minutes, stand hours, workouts and
notifications count as zero only on days the Watch recorded heart rate.
"Watch worn" means at least 10 hours of the day had a heart-rate reading.

## Sleep

* Kinds of night: **stages** (Watch, watchOS 9+), **asleep** (Watch or a
  sleep app, no stages) and **in bed only** (iPhone Bedtime / Sleep
  Schedule, which is schedule- and phone-use-based, not measured sleep).
* Per night, one source is used for "asleep" (Watch first) and one for
  "in bed" (iPhone first). Sources are never added together.
* Samples less than 90 minutes apart form an episode. Episodes centred between
  18:00 and 14:00 belong to the night ending that morning and are combined;
  a night totalling under 3 hours, or an episode centred 14:00 to 18:00, is a nap.
* Wrist-based staging is an estimate.

## Workouts

* The same session recorded by two apps (over 50% time overlap, same type) is
  kept once, preferring the Watch.
* Heart-rate zones use an age-based maximum (208 − 0.7 × age, Tanaka 2001).
  Individual maximum heart rates vary by 10 bpm or more, so zones are approximate.

## Findings

* **Year-on-year tiles:** last 365 days against the 365 before.
* **Trends:** Theil–Sen slope of monthly values with a Mann–Kendall test
  adjusted for month-to-month correlation. Only clear trends (p < 0.05 and a
  meaningful size) are described as rises or falls.
* **Chapters:** change points in monthly means (binary segmentation with a
  penalty), at most three.
* **Connections:** a fixed list of pairs. Slow drifts are removed with a
  29-day centred median, then a rank correlation is computed with an
  effective sample size for autocorrelation, corrected across pairs at a 5%
  false-discovery rate. The plain-language difference compares your top and
  bottom thirds with a block-bootstrap interval. These are associations, not causes.
* **Raised resting heart rate:** at least 3 bpm and two robust standard
  deviations above the median of the 60 days ending a week earlier, for 3+
  days (or 2+ days and 5+ bpm).

## Reference values used as context

| Topic | Reference |
|---|---|
| Activity minutes | WHO guidelines on physical activity (2020): 150–300 min moderate per week |
| Steps | Paluch et al., Lancet Public Health 2022 |
| Sleep duration | AASM / Sleep Research Society consensus (2015): 7+ hours for adults |
| VO₂ max medians | Kaminsky et al., Mayo Clin Proc 2015 (FRIEND registry, treadmill) |
| Maximum heart rate | Tanaka et al., J Am Coll Cardiol 2001 |

The VO₂ max medians are approximate values from that paper and Watch
estimates can differ from lab tests by several points; treat the comparison as rough.

## Known limits

* Not yet used: ECG recordings, workout GPS routes, clinical records,
  nutrition, cycle tracking.
* Types HealthLens does not know are imported and listed, but not analysed.
* Tested on synthetic exports modelled on Apple's format, not yet on a real one.
