"""Load Hevy set data and selected Apple Health recovery metrics."""
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd

MUSCLE_MAP = {
    "bench press": "chest", "incline bench": "chest", "chest fly": "chest",
    "push up": "chest", "dip": "chest",
    "overhead press": "shoulders", "shoulder press": "shoulders",
    "lateral raise": "shoulders", "face pull": "shoulders",
    "deadlift": "posterior", "romanian deadlift": "posterior",
    "hip thrust": "posterior", "good morning": "posterior",
    "squat": "quads", "leg press": "quads", "lunge": "quads",
    "leg extension": "quads",
    "lat pulldown": "back", "pull up": "back", "row": "back",
    "chin up": "back",
    "curl": "arms", "tricep": "arms", "pushdown": "arms",
    "skullcrusher": "arms",
}

REQUIRED_HEVY = {"start_time", "exercise_title", "set_type", "weight_kg", "reps"}


def muscle_of(exercise: str) -> str:
    name = str(exercise).lower()
    for key, muscle in MUSCLE_MAP.items():
        if key in name:
            return muscle
    return "other"


def load_hevy(csv_path: str | Path) -> pd.DataFrame:
    """Validate and tidy a Hevy set-level export.

    Missing RPE is allowed and maps to the neutral RPE-8 weighting.  Invalid
    numeric values fail loudly rather than being silently clipped.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_HEVY - set(df.columns)
    if missing:
        raise ValueError(f"Hevy export missing required columns: {sorted(missing)}")
    if "title" not in df:
        df["title"] = ""
    if "rpe" not in df:
        df["rpe"] = np.nan
    if df.duplicated().any():
        raise ValueError("Hevy export contains duplicate set rows")

    df["start_time"] = pd.to_datetime(df["start_time"], dayfirst=True,
                                      format="mixed", errors="raise")
    for col in ["weight_kg", "reps", "rpe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if (df["weight_kg"].dropna() < 0).any() or (df["reps"].dropna() < 0).any():
        raise ValueError("Hevy weight/reps cannot be negative")
    bad_rpe = df["rpe"].dropna().loc[lambda s: (s < 1) | (s > 10)]
    if len(bad_rpe):
        raise ValueError("Hevy RPE values must lie in [1, 10] when present")

    df["date"] = df["start_time"].dt.normalize()
    df["muscle"] = df["exercise_title"].map(muscle_of)
    df["working"] = ~df["set_type"].fillna("normal").eq("warmup")
    df["weight_kg"] = df["weight_kg"].fillna(0.0)
    df["reps"] = df["reps"].fillna(0.0)
    df["tonnage"] = np.where(df["working"], df["weight_kg"] * df["reps"], 0.0)
    rpe_for_weight = df["rpe"].fillna(8.0)
    df["effective_tonnage"] = df["tonnage"] * rpe_for_weight / 8.0
    return df[["date", "title", "exercise_title", "muscle", "set_type",
               "working", "weight_kg", "reps", "rpe", "tonnage",
               "effective_tonnage"]]


APPLE_TYPES = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierRestingHeartRate": "rhr",
}
ASLEEP_PREFIX = "HKCategoryValueSleepAnalysisAsleep"


def _merged_hours(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Union interval duration, preventing overlapping sleep double counts."""
    if not intervals:
        return float("nan")
    spans = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in spans:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    seconds = sum((end - start).total_seconds() for start, end in merged)
    return seconds / 3600.0


def load_apple_health(xml_path: str | Path) -> pd.DataFrame:
    """Daily HRV, resting HR and sleep from Apple Health `export.xml`.

    Parsing is streamed.  Sleep durations use the timezone-aware timestamps,
    overnight sleep is credited to the local calendar date on which it ends,
    and overlapping asleep intervals are merged before summation.
    """
    hrv, rhr, sleep_intervals = {}, {}, {}
    for _, elem in iterparse(str(xml_path), events=("end",)):
        if elem.tag != "Record":
            elem.clear()
            continue
        rtype = elem.get("type")
        start_text = elem.get("startDate")
        end_text = elem.get("endDate")
        if rtype in APPLE_TYPES and start_text:
            day = pd.Timestamp(start_text[:10])
            bucket = hrv if APPLE_TYPES[rtype] == "hrv" else rhr
            bucket.setdefault(day, []).append(float(elem.get("value")))
        elif (rtype == "HKCategoryTypeIdentifierSleepAnalysis"
              and str(elem.get("value", "")).startswith(ASLEEP_PREFIX)
              and start_text and end_text):
            # Full timestamps retain offsets for duration; the first 10 chars of
            # endDate preserve Apple's local calendar day for attribution.
            start = pd.Timestamp(start_text)
            end = pd.Timestamp(end_text)
            day = pd.Timestamp(end_text[:10])
            sleep_intervals.setdefault(day, []).append((start, end))
        elem.clear()
    days = sorted(set(hrv) | set(rhr) | set(sleep_intervals))
    return pd.DataFrame({
        "date": days,
        "hrv": [float(np.mean(hrv[d])) if d in hrv else np.nan for d in days],
        "rhr": [float(np.mean(rhr[d])) if d in rhr else np.nan for d in days],
        "sleep_hours": [_merged_hours(sleep_intervals.get(d, [])) for d in days],
    })
