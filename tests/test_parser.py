from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from healthlens.parser import (MISSING_TIME, NotAnExportError, SanitizingReader, import_export, locate_export,
                               parse_hk_dates)
from healthlens.store import HealthStore

WATCH_DEV = ("&lt;&lt;HKDevice: 0x283a1c5a0&gt;, name:Apple Watch, manufacturer:Apple Inc., model:Watch, "
             "hardware:Watch3,4, software:6.2&gt;")
PHONE_DEV = ("&lt;&lt;HKDevice: 0x2811&gt;, name:iPhone, manufacturer:Apple Inc., model:iPhone, "
             "hardware:iPhone10,1, software:13.5&gt;")

HANDCRAFTED = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!-- HealthKit Export Version: 11 -->
<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary)*)>
<!ATTLIST HealthData locale CDATA #REQUIRED>
]>
<HealthData locale="en_GB">
 <ExportDate value="2020-07-01 10:00:00 +0100"/>
 <Me HKCharacteristicTypeIdentifierDateOfBirth="1988-02-03" HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexFemale" HKCharacteristicTypeIdentifierBloodType="HKBloodTypeNotSet" HKCharacteristicTypeIdentifierFitzpatrickSkinType="HKFitzpatrickSkinTypeNotSet"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Sam’s Apple Watch" sourceVersion="6.2" device="{WATCH_DEV}" unit="count" creationDate="2020-06-30 08:10:00 +0100" startDate="2020-06-30 08:00:00 +0100" endDate="2020-06-30 08:09:00 +0100" value="900"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Sam’s iPhone" sourceVersion="13.5" device="{PHONE_DEV}" unit="count" creationDate="2020-06-30 08:12:00 +0100" startDate="2020-06-30 08:01:00 +0100" endDate="2020-06-30 08:10:00 +0100" value="850"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Sam’s Apple Watch" sourceVersion="6.2" device="{WATCH_DEV}" unit="count/min" creationDate="2020-06-30 08:05:00 +0100" startDate="2020-06-30 08:05:00 +0100" endDate="2020-06-30 08:05:00 +0100" value="96">
  <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" value="2"/>
 </Record>
 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Sam’s Apple Watch" sourceVersion="6.2" device="{WATCH_DEV}" unit="ms" creationDate="2020-06-30 22:00:00 +0100" startDate="2020-06-30 21:58:00 +0100" endDate="2020-06-30 21:59:00 +0100" value="48.5">
  <HeartRateVariabilityMetadataList>
   <InstantaneousBeatsPerMinute bpm="61" time="9:58:01.11 PM"/>
   <InstantaneousBeatsPerMinute bpm="63" time="9:58:02.10 PM"/>
  </HeartRateVariabilityMetadataList>
 </Record>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Clock" sourceVersion="50" device="{PHONE_DEV}" creationDate="2020-06-30 07:00:00 +0100" startDate="2020-06-29 23:10:00 +0100" endDate="2020-06-30 06:55:00 +0100" value="HKCategoryValueSleepAnalysisInBed">
  <MetadataEntry key="HKTimeZone" value="Europe/London"/>
 </Record>
 <Record type="HKQuantityTypeIdentifierBodyMass" sourceName="Health" sourceVersion="13.5" unit="lb" creationDate="2020-06-30 07:30:00 +0100" startDate="2020-06-30 07:30:00 +0100" endDate="2020-06-30 07:30:00 +0100" value="150.5">
  <MetadataEntry key="HKWasUserEntered" value="1"/>
 </Record>
 <Record type="HKQuantityTypeIdentifierBloodPressureSystolic" sourceName="Health" sourceVersion="13.5" unit="mmHg" creationDate="2020-06-30 09:00:00 +0100" startDate="2020-06-30 09:00:00 +0100" endDate="2020-06-30 09:00:00 +0100" value="118"/>
 <Correlation type="HKCorrelationTypeIdentifierBloodPressure" sourceName="Health" sourceVersion="13.5" creationDate="2020-06-30 09:00:00 +0100" startDate="2020-06-30 09:00:00 +0100" endDate="2020-06-30 09:00:00 +0100">
  <Record type="HKQuantityTypeIdentifierBloodPressureSystolic" sourceName="Health" sourceVersion="13.5" unit="mmHg" creationDate="2020-06-30 09:00:00 +0100" startDate="2020-06-30 09:00:00 +0100" endDate="2020-06-30 09:00:00 +0100" value="118"/>
  <Record type="HKQuantityTypeIdentifierBloodPressureDiastolic" sourceName="Health" sourceVersion="13.5" unit="mmHg" creationDate="2020-06-30 09:00:00 +0100" startDate="2020-06-30 09:00:00 +0100" endDate="2020-06-30 09:00:00 +0100" value="76"/>
 </Correlation>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30.5" durationUnit="min" totalDistance="3.1" totalDistanceUnit="mi" totalEnergyBurned="320" totalEnergyBurnedUnit="kcal" sourceName="Sam’s Apple Watch" sourceVersion="6.2" device="{WATCH_DEV}" creationDate="2020-06-30 19:01:00 +0100" startDate="2020-06-30 18:30:00 +0100" endDate="2020-06-30 19:00:30 +0100">
  <MetadataEntry key="HKIndoorWorkout" value="0"/>
  <MetadataEntry key="HKElevationAscended" value="4500 cm"/>
  <MetadataEntry key="HKWeatherTemperature" value="68 degF"/>
  <MetadataEntry key="HKWeatherHumidity" value="5500 %"/>
  <WorkoutEvent type="HKWorkoutEventTypePause" date="2020-06-30 18:40:00 +0100"/>
  <WorkoutRoute sourceName="Sam’s Apple Watch" sourceVersion="6.2" creationDate="2020-06-30 19:02:00 +0100" startDate="2020-06-30 18:30:00 +0100" endDate="2020-06-30 19:00:30 +0100">
   <MetadataEntry key="HKMetadataKeySyncVersion" value="2"/>
   <FileReference path="/workout-routes/route_2020-06-30_6.30pm.gpx"/>
  </WorkoutRoute>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeCycling" duration="45" durationUnit="min" sourceName="Sam’s Apple Watch" sourceVersion="9.0" device="{WATCH_DEV}" creationDate="2020-06-30 21:00:00 +0100" startDate="2020-06-30 20:00:00 +0100" endDate="2020-06-30 20:45:00 +0100">
  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" startDate="2020-06-30 20:00:00 +0100" endDate="2020-06-30 20:45:00 +0100" sum="410" unit="kcal"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierDistanceCycling" startDate="2020-06-30 20:00:00 +0100" endDate="2020-06-30 20:45:00 +0100" sum="18.2" unit="km"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierHeartRate" startDate="2020-06-30 20:00:00 +0100" endDate="2020-06-30 20:45:00 +0100" average="138.5" minimum="95" maximum="171" unit="count/min"/>
 </Workout>
 <ActivitySummary dateComponents="2020-06-30" activeEnergyBurned="612.5" activeEnergyBurnedGoal="500" activeEnergyBurnedUnit="kcal" appleExerciseTime="62" appleExerciseTimeGoal="30" appleStandHours="11" appleStandHoursGoal="12"/>
 <ActivitySummary dateComponents="2020-06-29" activeEnergyBurned="0" activeEnergyBurnedGoal="500" activeEnergyBurnedUnit="kcal" appleExerciseTime="0" appleStandHours="0"/>
 <Audiogram type="HKDataTypeIdentifierAudiogram" sourceName="Mimi" startDate="2020-06-30 09:00:00 +0100" endDate="2020-06-30 09:00:00 +0100"/>
</HealthData>
"""


def _ts(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S %z").timestamp())


# ---------------------------------------------------------------------------
def test_parse_hk_dates_handles_offsets_and_fallbacks():
    values = ["2020-06-30 08:00:00 +0100", "2020-12-31 23:59:59 -0800", "2021-03-14 12:00:00 +0530",
              "1999-01-01 00:00:00 +0000", "2020-06-30T08:00:00+01:00", "", None, "garbage"]
    utc, off = parse_hk_dates(values)
    assert utc[0] == _ts(values[0]) and off[0] == 60
    assert utc[1] == _ts(values[1]) and off[1] == -480
    assert utc[2] == _ts(values[2]) and off[2] == 330
    assert utc[3] == _ts(values[3]) and off[3] == 0
    assert utc[4] == _ts(values[0]) and off[4] == 60          # ISO fallback
    assert utc[5] == MISSING_TIME and utc[6] == MISSING_TIME and utc[7] == MISSING_TIME


def test_parse_hk_dates_random_roundtrip():
    rng = np.random.default_rng(0)
    secs = rng.integers(0, 2_000_000_000, 500)
    offs = rng.choice([-480, -300, 0, 60, 330, 345, 600], 500)
    texts = []
    for s, o in zip(secs.tolist(), offs.tolist()):
        tz = timezone(pd.Timedelta(minutes=o).to_pytimedelta())
        texts.append(datetime.fromtimestamp(s, tz).strftime("%Y-%m-%d %H:%M:%S %z"))
    utc, off = parse_hk_dates(texts)
    assert (utc == secs).all() and (off == offs).all()


class _Trickle:
    """Raw stream returning tiny chunks to exercise chunk-boundary handling."""

    def __init__(self, data: bytes, size: int = 7):
        self.buf = io.BytesIO(data)
        self.size = size

    def read(self, n=-1):
        return self.buf.read(self.size)

    def close(self):
        pass


def test_sanitizer_strips_dtd_and_illegal_characters():
    raw = ('<?xml version="1.0"?>\n<!DOCTYPE HealthData [ <!ELEMENT broken ( ]>\n'
           '<HealthData locale="x"><Record a="bad\x0bchar &#x0B; &#11; &#x1F600; fine"/></HealthData>').encode()
    reader = SanitizingReader(_Trickle(raw))
    out = b""
    while True:
        chunk = reader.read(5)
        if not chunk:
            break
        out += chunk
    text = out.decode()
    assert "DOCTYPE" not in text
    assert "\x0b" not in text and "&#x0B;" not in text and "&#11;" not in text
    assert "&#x1F600;" in text and "fine" in text


def test_not_an_export(tmp_path):
    p = tmp_path / "random.xml"
    p.write_text("<?xml version='1.0'?><Something/>")
    with pytest.raises(NotAnExportError):
        import_export(p, tmp_path / "cache")


def test_handcrafted_import(tmp_path):
    xml = tmp_path / "export.xml"
    xml.write_text(HANDCRAFTED, encoding="utf-8")
    meta = import_export(xml, tmp_path / "cache")
    store = HealthStore(tmp_path / "cache")

    assert meta["export_version"] == 11
    assert meta["locale"] == "en_GB"
    assert meta["profile"]["DateOfBirth"] == "1988-02-03"
    assert meta["other_elements"] == {"Audiogram": 1}

    steps = store.records("StepCount")
    assert len(steps) == 2
    assert set(steps["kind"]) == {"watch", "phone"}
    assert steps["start"].iloc[0] == _ts("2020-06-30 08:00:00 +0100")
    assert (steps["start_off"] == 60).all()

    hr = store.records("HeartRate")
    assert hr["hr_ctx"].tolist() == [2]
    hrv = store.records("HeartRateVariabilitySDNN")
    assert hrv["value"].tolist() == [48.5]

    sleep = store.records("SleepAnalysis")
    assert sleep["cat"].tolist() == ["SleepAnalysisInBed"]
    assert sleep["tz"].tolist() == ["Europe/London"]
    assert sleep["kind"].tolist() == ["phone"]       # "Clock" app = iPhone Bedtime

    weight = store.values("BodyMass", "kg")
    assert weight["value"].iloc[0] == pytest.approx(150.5 * 0.45359237)
    assert bool(weight["user_entered"].iloc[0])

    # correlation member duplicated at top level must not be double counted;
    # the diastolic value only exists inside the correlation and must be kept
    assert len(store.records("BloodPressureSystolic")) == 1
    assert store.records("BloodPressureDiastolic")["value"].tolist() == [76.0]

    w = store.workouts().sort_values("start").reset_index(drop=True)
    assert w["activity"].tolist() == ["Running", "Cycling"]
    run, ride = w.iloc[0], w.iloc[1]
    assert run["distance_km"] == pytest.approx(3.1 * 1.609344)
    assert run["energy_kcal"] == pytest.approx(320)
    assert run["elevation_m"] == pytest.approx(45.0)
    assert run["temperature_c"] == pytest.approx(20.0)
    assert run["humidity_pct"] == pytest.approx(55.0)
    assert run["route"] == "/workout-routes/route_2020-06-30_6.30pm.gpx"
    assert run["pauses"] == 1 and not bool(run["indoor"])
    assert ride["distance_km"] == pytest.approx(18.2)          # from WorkoutStatistics (iOS 16+ format)
    assert ride["energy_kcal"] == pytest.approx(410)
    assert ride["avg_hr"] == pytest.approx(138.5) and ride["max_hr"] == 171

    act = store.activity_summary()
    assert len(act) == 2
    row = act[act["date"] == pd.Timestamp("2020-06-30")].iloc[0]
    assert row["exercise_min"] == 62 and row["stand_goal_h"] == 12

    src = store.sources.set_index("name")
    assert src.loc["Sam’s Apple Watch", "friendly"] == "Apple Watch Series 3 42mm"
    assert src.loc["Sam’s iPhone", "friendly"] == "iPhone 8"
    assert src.loc["Health", "kind"] == "manual"


def test_localised_zip_and_skip_reimport(tmp_path):
    z = tmp_path / "Exportieren.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("apple_health_export/导出.xml", HANDCRAFTED)
        zf.writestr("apple_health_export/导出_cda.xml", "<ClinicalDocument/>")
        zf.writestr("apple_health_export/electrocardiograms/ecg_2020-06-30.csv", "Name,Sam\n")
    src = locate_export(z)
    assert src.xml_name.endswith("导出.xml")
    meta = import_export(z, tmp_path / "cache")
    assert meta["n_workouts"] == 2 and meta["ecg_files"] == 1
    again = import_export(z, tmp_path / "cache")
    assert again.get("skipped") is True


def test_repair_mode_recovers_invalid_utf8(tmp_path):
    bad = HANDCRAFTED.encode("utf-8").replace("Sam’s iPhone".encode("utf-8"), b"Sam\xff\xfes iPhone")
    p = tmp_path / "export.xml"
    p.write_bytes(bad)
    meta = import_export(p, tmp_path / "cache")
    assert meta["repair_mode"] is True
    store = HealthStore(tmp_path / "cache")
    assert len(store.records("StepCount")) == 2


def test_failed_import_keeps_previous_cache(tmp_path):
    good = tmp_path / "export.xml"
    good.write_text(HANDCRAFTED, encoding="utf-8")
    import_export(good, tmp_path / "cache")
    broken = tmp_path / "broken.xml"
    broken.write_text(HANDCRAFTED[: len(HANDCRAFTED) // 2] + "<<<", encoding="utf-8")
    with pytest.raises(NotAnExportError):
        import_export(broken, tmp_path / "cache")
    meta = json.loads((tmp_path / "cache" / "meta.json").read_text())
    assert meta["n_workouts"] == 2


def test_synthetic_roundtrip(small_store, small_export):
    _, truth, cfg = small_export
    meta = small_store.meta
    assert meta["n_records"] > 10_000
    for t in ("StepCount", "HeartRate", "SleepAnalysis", "RestingHeartRate", "VO2Max", "ActiveEnergyBurned"):
        assert small_store.has(t), t
    rhr = small_store.records("RestingHeartRate")
    assert len(rhr) == int(truth["worn"].sum())
    stages = set(small_store.records("SleepAnalysis")["cat"].dropna())
    # spans watchOS 8 (unspecified 'asleep') and watchOS 9 (stages)
    assert {"SleepAnalysisAsleepUnspecified", "SleepAnalysisAsleepCore", "SleepAnalysisAsleepDeep",
            "SleepAnalysisAsleepREM", "SleepAnalysisInBed"} <= stages
    # DST change on 2022-11-06 shows up as two different offsets
    assert set(small_store.records("StepCount")["start_off"].unique()) == {-240, -300}
