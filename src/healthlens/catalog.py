"""What we know about HealthKit data types.

Each :class:`Metric` describes one column of the daily table: which HealthKit
type feeds it, how a day's samples are combined, the canonical unit, and which
direction is generally "better" (used only to choose neutral wording such as
"improved" vs "declined"; it is never a medical judgement).

Types not listed here are still imported and summarised generically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

QUANTITY_PREFIX = "HKQuantityTypeIdentifier"
CATEGORY_PREFIX = "HKCategoryTypeIdentifier"
OTHER_PREFIXES = ("HKDataType", "HKCorrelationTypeIdentifier", "HKWorkoutTypeIdentifier")
CATEGORY_VALUE_PREFIX = "HKCategoryValue"
WORKOUT_PREFIX = "HKWorkoutActivityType"


def short_type(hk_identifier: str) -> str:
    """'HKQuantityTypeIdentifierStepCount' -> 'StepCount'."""
    for prefix in (QUANTITY_PREFIX, CATEGORY_PREFIX) + OTHER_PREFIXES:
        if hk_identifier.startswith(prefix):
            return hk_identifier[len(prefix):]
    return hk_identifier


def type_kind(hk_identifier: str) -> str:
    if hk_identifier.startswith(QUANTITY_PREFIX):
        return "quantity"
    if hk_identifier.startswith(CATEGORY_PREFIX):
        return "category"
    return "other"


def short_category_value(value: str) -> str:
    """'HKCategoryValueSleepAnalysisAsleepCore' -> 'SleepAnalysisAsleepCore'."""
    if value.startswith(CATEGORY_VALUE_PREFIX):
        return value[len(CATEGORY_VALUE_PREFIX):]
    return value


def short_workout(activity: str) -> str:
    """'HKWorkoutActivityTypeRunning' -> 'Running'."""
    if activity.startswith(WORKOUT_PREFIX):
        return activity[len(WORKOUT_PREFIX):]
    return activity


@dataclass(frozen=True)
class Metric:
    key: str             # column name in the daily table
    hk: str              # short HealthKit type name feeding it ("" for derived metrics)
    label: str
    domain: str          # activity | heart | sleep | body | mobility | respiratory | hearing | mind | workouts
    agg: str             # sum | mean | last | max | min
    unit: str            # canonical unit (see units.py)
    better: str = "neutral"  # higher | lower | neutral | range
    decimals: int = 0
    dedup: bool = False  # cumulative type that iPhone + Watch both record -> source-priority merge
    description: str = ""


METRICS: List[Metric] = [
    # ---------------- Activity ----------------
    Metric("steps", "StepCount", "Steps", "activity", "sum", "count", "higher", 0, True,
           "Steps counted by your Watch and iPhone, merged so the same step is not counted twice."),
    Metric("distance_km", "DistanceWalkingRunning", "Walking + running distance", "activity", "sum", "km",
           "higher", 2, True, "Distance covered on foot."),
    Metric("flights", "FlightsClimbed", "Flights climbed", "activity", "sum", "count", "higher", 0, True,
           "Floors climbed (about 3 m / 10 ft of elevation each)."),
    Metric("active_kcal", "ActiveEnergyBurned", "Active energy", "activity", "sum", "kcal", "higher", 0, True,
           "Calories burned through movement, on top of your resting burn (the Move ring)."),
    Metric("basal_kcal", "BasalEnergyBurned", "Resting energy", "activity", "sum", "kcal", "neutral", 0, True,
           "Calories your body burns at rest; mostly reflects body size, not behaviour."),
    Metric("exercise_min", "AppleExerciseTime", "Exercise minutes", "activity", "sum", "min", "higher", 0, True,
           "Minutes of brisk activity or more (the Exercise ring)."),
    Metric("stand_min", "AppleStandTime", "Stand minutes", "activity", "sum", "min", "higher", 0, True,
           "Minutes the Watch detected you standing and moving (watchOS 7 and later)."),
    Metric("move_min", "AppleMoveTime", "Move minutes", "activity", "sum", "min", "higher", 0, True,
           "Move-time goal minutes (used instead of calories in some Watch setups)."),
    Metric("cycling_km", "DistanceCycling", "Cycling distance", "activity", "sum", "km", "higher", 1, True, ""),
    Metric("swimming_m", "DistanceSwimming", "Swimming distance", "activity", "sum", "m", "higher", 0, True, ""),
    Metric("swim_strokes", "SwimmingStrokeCount", "Swimming strokes", "activity", "sum", "count", "higher", 0, True, ""),
    Metric("wheelchair_pushes", "PushCount", "Wheelchair pushes", "activity", "sum", "count", "higher", 0, True, ""),
    Metric("daylight_min", "TimeInDaylight", "Time in daylight", "activity", "sum", "min", "higher", 0, True,
           "Minutes the Watch's light sensor saw daylight (newer Watches only)."),
    Metric("physical_effort", "PhysicalEffort", "Physical effort", "activity", "mean", "METs", "neutral", 1, False,
           "Average intensity in METs (watchOS 10 and later)."),
    # ---------------- Heart & fitness ----------------
    Metric("resting_hr", "RestingHeartRate", "Resting heart rate", "heart", "mean", "bpm", "lower", 0, False,
           "Your heart rate while calm and still, estimated by the Watch once a day. Lower usually reflects better "
           "cardiovascular fitness; short-term rises often accompany illness, poor sleep, stress, alcohol or heat."),
    Metric("walking_hr", "WalkingHeartRateAverage", "Walking heart rate", "heart", "mean", "bpm", "lower", 0, False,
           "Average heart rate during everyday walking; tends to fall as fitness improves."),
    Metric("hrv", "HeartRateVariabilitySDNN", "Heart-rate variability (SDNN)", "heart", "mean", "ms", "higher", 0,
           False, "Variation between heartbeats (SDNN, the measure Apple uses). Very individual: compare with your "
                  "own baseline rather than with other people. Higher than your norm usually means well recovered."),
    Metric("vo2max", "VO2Max", "Cardio fitness (VO₂ max)", "heart", "mean", "mL/kg/min", "higher", 1, False,
           "Estimated maximum oxygen uptake from outdoor walks, runs and hikes. One of the strongest everyday "
           "markers of long-term health. Watch estimates can differ from a lab test by several points."),
    Metric("hr_recovery", "HeartRateRecoveryOneMinute", "Heart-rate recovery (1 min)", "heart", "mean", "bpm",
           "higher", 0, False, "How far your heart rate drops in the minute after hard exercise."),
    Metric("afib_burden", "AtrialFibrillationBurden", "AFib burden", "heart", "mean", "%", "lower", 1, False,
           "Share of time with signs of AFib (only for people diagnosed with AFib who enabled AFib History)."),
    # ---------------- Respiratory ----------------
    Metric("spo2", "OxygenSaturation", "Blood oxygen", "respiratory", "mean", "%", "range", 1, False,
           "Blood-oxygen readings (Series 6 and later). Apple states these are for wellness, not medical use."),
    Metric("resp_rate", "RespiratoryRate", "Respiratory rate (sleep)", "respiratory", "mean", "br/min", "range", 1,
           False, "Breaths per minute measured while you sleep (watchOS 8 and later, only on nights you wore the Watch)."),
    Metric("breathing_disturbances", "AppleSleepingBreathingDisturbances", "Breathing disturbances", "respiratory",
           "mean", "count", "lower", 1, False, "Nightly breathing-disturbance estimate (newer Watches only)."),
    # ---------------- Body ----------------
    Metric("weight_kg", "BodyMass", "Weight", "body", "last", "kg", "neutral", 1, False, ""),
    Metric("bmi", "BodyMassIndex", "BMI", "body", "last", "kg/m²", "neutral", 1, False, ""),
    Metric("body_fat_pct", "BodyFatPercentage", "Body fat", "body", "last", "%", "neutral", 1, False, ""),
    Metric("lean_mass_kg", "LeanBodyMass", "Lean body mass", "body", "last", "kg", "neutral", 1, False, ""),
    Metric("height_cm", "Height", "Height", "body", "last", "cm", "neutral", 0, False, ""),
    Metric("wrist_temp_c", "AppleSleepingWristTemperature", "Sleeping wrist temperature", "body", "mean", "degC",
           "neutral", 2, False, "Wrist temperature during sleep (Series 8 and later)."),
    Metric("body_temp_c", "BodyTemperature", "Body temperature", "body", "mean", "degC", "neutral", 1, False, ""),
    Metric("bp_systolic", "BloodPressureSystolic", "Blood pressure (systolic)", "body", "mean", "mmHg", "lower", 0,
           False, ""),
    Metric("bp_diastolic", "BloodPressureDiastolic", "Blood pressure (diastolic)", "body", "mean", "mmHg", "lower",
           0, False, ""),
    # ---------------- Mobility (mostly iPhone) ----------------
    Metric("walking_speed_kmh", "WalkingSpeed", "Walking speed", "mobility", "mean", "km/h", "higher", 2, False,
           "Typical walking speed measured by your iPhone (iOS 14+)."),
    Metric("step_length_cm", "WalkingStepLength", "Step length", "mobility", "mean", "cm", "higher", 0, False, ""),
    Metric("walking_asymmetry_pct", "WalkingAsymmetryPercentage", "Walking asymmetry", "mobility", "mean", "%",
           "lower", 1, False, "How often one step is noticeably faster or slower than the other."),
    Metric("double_support_pct", "WalkingDoubleSupportPercentage", "Double-support time", "mobility", "mean", "%",
           "lower", 1, False, "Share of walking time with both feet on the ground."),
    Metric("stair_ascent_speed", "StairAscentSpeed", "Stair speed (up)", "mobility", "mean", "m/s", "higher", 2,
           False, ""),
    Metric("stair_descent_speed", "StairDescentSpeed", "Stair speed (down)", "mobility", "mean", "m/s", "higher", 2,
           False, ""),
    Metric("walking_steadiness_pct", "AppleWalkingSteadiness", "Walking steadiness", "mobility", "mean", "%",
           "higher", 0, False, ""),
    Metric("six_min_walk_m", "SixMinuteWalkTestDistance", "Six-minute walk estimate", "mobility", "last", "m",
           "higher", 0, False, ""),
    # ---------------- Running form (watchOS 9+) ----------------
    Metric("running_speed_kmh", "RunningSpeed", "Running speed", "workouts", "mean", "km/h", "higher", 1, False, ""),
    Metric("running_power_w", "RunningPower", "Running power", "workouts", "mean", "W", "neutral", 0, False, ""),
    Metric("running_stride_m", "RunningStrideLength", "Running stride length", "workouts", "mean", "m", "neutral", 2,
           False, ""),
    Metric("running_vo_cm", "RunningVerticalOscillation", "Vertical oscillation", "workouts", "mean", "cm", "lower",
           1, False, ""),
    Metric("running_gct_ms", "RunningGroundContactTime", "Ground contact time", "workouts", "mean", "ms", "lower", 0,
           False, ""),
    # ---------------- Hearing ----------------
    Metric("env_audio_db", "EnvironmentalAudioExposure", "Environmental noise", "hearing", "mean", "dBASPL", "lower",
           0, False, "Average sound level around you (Series 4+ with watchOS 6+)."),
    Metric("headphone_audio_db", "HeadphoneAudioExposure", "Headphone volume", "hearing", "mean", "dBASPL", "lower",
           0, False, "Average headphone listening level measured by your iPhone."),
]

# Derived daily columns that do not come straight from one HealthKit type.
DERIVED_METRICS: List[Metric] = [
    Metric("hr_mean", "HeartRate", "Average heart rate", "heart", "mean", "bpm", "neutral", 0, False,
           "Average of hourly heart-rate means while the Watch was worn."),
    Metric("hr_min", "HeartRate", "Lowest heart rate", "heart", "min", "bpm", "neutral", 0, False, ""),
    Metric("hr_max", "HeartRate", "Highest heart rate", "heart", "max", "bpm", "neutral", 0, False, ""),
    Metric("watch_wear_h", "", "Watch wear time (est.)", "data", "sum", "h", "neutral", 1, False,
           "Hours of the day with at least one Watch heart-rate reading: a proxy for when you wore it."),
    Metric("stand_hours", "AppleStandHour", "Stand hours", "activity", "sum", "count", "higher", 0, False,
           "Hours in which you stood and moved for at least a minute (the Stand ring)."),
    Metric("mindful_min", "MindfulSession", "Mindful minutes", "mind", "sum", "min", "higher", 0, False, ""),
    Metric("sleep_h", "", "Sleep (measured)", "sleep", "sum", "h", "range", 2, False,
           "Time asleep measured by the Watch or a sleep app."),
    Metric("in_bed_h", "", "Time in bed", "sleep", "sum", "h", "range", 2, False,
           "Time in bed, mostly from the iPhone's Bedtime / Sleep Schedule. Not the same as time asleep."),
    Metric("sleep_any_h", "", "Sleep (best available)", "sleep", "sum", "h", "range", 2, False,
           "Measured sleep where available, otherwise time in bed."),
    Metric("sleep_efficiency_pct", "", "Sleep efficiency", "sleep", "mean", "%", "higher", 0, False,
           "Share of time in bed actually spent asleep (nights with both measures)."),
    Metric("deep_h", "", "Deep sleep", "sleep", "sum", "h", "neutral", 2, False, ""),
    Metric("rem_h", "", "REM sleep", "sleep", "sum", "h", "neutral", 2, False, ""),
    Metric("core_h", "", "Core sleep", "sleep", "sum", "h", "neutral", 2, False, ""),
    Metric("awake_h", "", "Awake in night", "sleep", "sum", "h", "lower", 2, False, ""),
    Metric("bedtime_h", "", "Bedtime", "sleep", "mean", "clock", "neutral", 2, False,
           "Hours after midnight (e.g. 23.5 = 11:30 pm, 24.5 = 12:30 am)."),
    Metric("wake_h", "", "Wake time", "sleep", "mean", "clock", "neutral", 2, False, ""),
    Metric("sleep_hr", "", "Sleeping heart rate", "sleep", "mean", "bpm", "lower", 0, False,
           "Average heart rate while asleep, on nights you wore the Watch."),
    Metric("nap_h", "", "Naps", "sleep", "sum", "h", "neutral", 2, False, ""),
    Metric("workouts_n", "", "Workouts", "workouts", "sum", "count", "higher", 0, False, ""),
    Metric("workout_min", "", "Workout minutes", "workouts", "sum", "min", "higher", 0, False, ""),
    Metric("workout_kcal", "", "Workout energy", "workouts", "sum", "kcal", "higher", 0, False, ""),
    Metric("high_hr_events", "HighHeartRateEvent", "High heart-rate notifications", "heart", "sum", "events",
           "neutral", 0, False, ""),
    Metric("low_hr_events", "LowHeartRateEvent", "Low heart-rate notifications", "heart", "sum", "events",
           "neutral", 0, False, ""),
    Metric("irregular_rhythm_events", "IrregularHeartRhythmEvent", "Irregular rhythm notifications", "heart", "sum",
           "events", "neutral", 0, False, ""),
    Metric("move_goal_kcal", "", "Move goal", "activity", "last", "kcal", "neutral", 0, False, ""),
    Metric("exercise_goal_min", "", "Exercise goal", "activity", "last", "min", "neutral", 0, False, ""),
    Metric("stand_goal_h", "", "Stand goal", "activity", "last", "count", "neutral", 0, False, ""),
    Metric("move_ring_pct", "", "Move ring", "activity", "mean", "%", "higher", 0, False, ""),
    Metric("exercise_ring_pct", "", "Exercise ring", "activity", "mean", "%", "higher", 0, False, ""),
    Metric("stand_ring_pct", "", "Stand ring", "activity", "mean", "%", "higher", 0, False, ""),
]

ALL_METRICS: Dict[str, Metric] = {m.key: m for m in METRICS + DERIVED_METRICS}
BY_HK: Dict[str, Metric] = {m.hk: m for m in METRICS}

# Category types that are counted as events per day (column name -> HK short type).
EVENT_TYPES = {
    "high_hr_events": "HighHeartRateEvent",
    "low_hr_events": "LowHeartRateEvent",
    "irregular_rhythm_events": "IrregularHeartRhythmEvent",
}

DOMAIN_LABELS = {
    "activity": "Activity",
    "heart": "Heart & fitness",
    "sleep": "Sleep",
    "workouts": "Workouts",
    "body": "Body",
    "mobility": "Mobility",
    "respiratory": "Breathing & oxygen",
    "hearing": "Hearing",
    "mind": "Mindfulness",
    "data": "Data coverage",
}


def metric(key: str) -> Optional[Metric]:
    return ALL_METRICS.get(key)


def label_for_type(short: str) -> str:
    """Readable label for an arbitrary HealthKit short type name."""
    m = BY_HK.get(short)
    if m:
        return m.label
    # 'DietaryEnergyConsumed' -> 'Dietary energy consumed'
    out = []
    for i, ch in enumerate(short):
        if ch.isupper() and i and not short[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    text = "".join(out).strip()
    return text[:1].upper() + text[1:].lower()


# Sleep stage values (short category values) -> normalised stage name.
SLEEP_STAGE = {
    "SleepAnalysisInBed": "inbed",
    "SleepAnalysisAsleep": "asleep",
    "SleepAnalysisAsleepUnspecified": "asleep",
    "SleepAnalysisAwake": "awake",
    "SleepAnalysisAsleepCore": "core",
    "SleepAnalysisAsleepDeep": "deep",
    "SleepAnalysisAsleepREM": "rem",
    # very old exports used integer codes
    "0": "inbed", "1": "asleep", "2": "awake", "3": "core", "4": "deep", "5": "rem",
}
ASLEEP_STAGES = ("asleep", "core", "deep", "rem")

WORKOUT_LABELS = {
    "TraditionalStrengthTraining": "Strength training",
    "FunctionalStrengthTraining": "Functional strength",
    "HighIntensityIntervalTraining": "HIIT",
    "MixedCardio": "Mixed cardio",
    "MixedMetabolicCardioTraining": "Mixed cardio",
    "CoreTraining": "Core training",
    "CrossTraining": "Cross training",
    "StairClimbing": "Stair climbing",
    "Elliptical": "Elliptical",
    "Rowing": "Rowing",
    "MindAndBody": "Mind & body",
    "Pilates": "Pilates",
    "Yoga": "Yoga",
    "Walking": "Walking",
    "Running": "Running",
    "Cycling": "Cycling",
    "Swimming": "Swimming",
    "Hiking": "Hiking",
    "Dance": "Dance",
    "SocialDance": "Dance",
    "CardioDance": "Dance",
    "Cooldown": "Cooldown",
    "Other": "Other",
    "Tennis": "Tennis",
    "Badminton": "Badminton",
    "Cricket": "Cricket",
    "Soccer": "Football (soccer)",
    "Basketball": "Basketball",
    "Golf": "Golf",
    "Climbing": "Climbing",
    "Boxing": "Boxing",
    "MartialArts": "Martial arts",
    "Kickboxing": "Kickboxing",
    "Flexibility": "Flexibility",
    "Barre": "Barre",
    "SnowSports": "Snow sports",
    "DownhillSkiing": "Downhill skiing",
    "CrossCountrySkiing": "Cross-country skiing",
    "Snowboarding": "Snowboarding",
    "WheelchairWalkPace": "Wheelchair (walk pace)",
    "WheelchairRunPace": "Wheelchair (run pace)",
    "HandCycling": "Hand cycling",
    "Paddling": "Paddling",
    "TableTennis": "Table tennis",
    "Squash": "Squash",
    "Volleyball": "Volleyball",
    "Rugby": "Rugby",
    "Hockey": "Hockey",
    "Skating": "Skating",
    "SwimBikeRun": "Multisport",
    "Transition": "Transition",
}

# Workouts whose distance makes a pace meaningful.
PACE_WORKOUTS = ("Running", "Walking", "Hiking", "Cycling", "Swimming", "WheelchairRunPace", "WheelchairWalkPace")


def workout_label(activity: str) -> str:
    if activity in WORKOUT_LABELS:
        return WORKOUT_LABELS[activity]
    return label_for_type(activity)
