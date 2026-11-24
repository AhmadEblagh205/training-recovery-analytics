"""Training analytics from the set log."""
import numpy as np
import pandas as pd


def epley_1rm(weight: float, reps: float) -> float:
    """Epley estimated 1RM: weight * (1 + reps / 30)."""
    return weight * (1 + reps / 30.0)


def e1rm_series(workouts: pd.DataFrame, exercise: str) -> pd.Series:
    """Best estimated 1RM per session for one exercise."""
    w = workouts[(workouts.exercise_title == exercise) & workouts.working]
    if w.empty:
        return pd.Series(dtype=float)
    est = w.assign(e1rm=epley_1rm(w.weight_kg, w.reps))
    return est.groupby("date")["e1rm"].max()


def personal_records(e1rm: pd.Series) -> pd.Series:
    """Sessions where estimated 1RM strictly exceeds every prior session."""
    if e1rm.empty:
        return e1rm
    prior_best = e1rm.cummax().shift(1)
    is_record = prior_best.isna() | e1rm.gt(prior_best)
    return e1rm[is_record]


def weekly_volume(workouts: pd.DataFrame) -> pd.DataFrame:
    """Tonnage per muscle per ISO-style weekly period."""
    w = workouts[workouts.working].copy()
    w["week"] = w["date"].dt.to_period("W").dt.start_time
    return (w.pivot_table(index="week", columns="muscle",
                          values="tonnage", aggfunc="sum")
            .fillna(0.0))


def daily_load(workouts: pd.DataFrame, all_days: pd.DatetimeIndex,
               metric: str = "tonnage") -> pd.Series:
    """Daily training load.

    `effective_tonnage` is the repository's explicit RPE-weighted heuristic,
    not a validated physiological dose measure.
    """
    col = metric if metric in workouts.columns else "tonnage"
    load = workouts[workouts.working].groupby("date")[col].sum()
    return load.reindex(all_days, fill_value=0.0)


def acwr(load: pd.Series, acute: int = 7, chronic: int = 28) -> pd.Series:
    """EWMA recent-to-longer-horizon workload ratio.

    The historical name ACWR is retained for compatibility.  In this repo it
    is a *descriptive load-balance diagnostic only*.  It is not interpreted as
    an injury-risk probability and no universal 'safe' or 'sweet-spot' range
    is asserted.
    """
    a = load.ewm(span=acute, adjust=False).mean()
    c = load.ewm(span=chronic, adjust=False).mean()
    return (a / c.replace(0, np.nan)).rename("acwr")
