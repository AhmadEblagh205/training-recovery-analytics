"""Transparent planning heuristics layered on the analytics.

The output is deliberately framed as an inspectable planning suggestion, not
medical advice, injury prediction or a validated coaching prescription.
"""
import pandas as pd


def volume_deficit(workouts: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Last-7-day tonnage relative to the *preceding* 28-day weekly mean."""
    w = workouts[workouts.working & (workouts.date <= asof)]
    last7 = w[w.date > asof - pd.Timedelta(days=7)].groupby("muscle").tonnage.sum()
    previous28 = w[(w.date <= asof - pd.Timedelta(days=7))
                   & (w.date > asof - pd.Timedelta(days=35))]
    baseline = previous28.groupby("muscle").tonnage.sum() / 4
    ratio = (last7 / baseline.replace(0, pd.NA)).reindex(baseline.index)
    return ratio.fillna(0.0).sort_values()


def recommend(readiness: float, acwr_today: float,
              deficit: pd.Series) -> dict:
    """Illustrative rules for session planning, with reasons exposed."""
    reasons = []
    if pd.isna(acwr_today):
        acwr_today = 1.0
    if readiness < -1.0:
        session, mult = "rest or mobility", 0.0
        reasons.append(f"readiness heuristic {readiness:+.1f} sd is unusually "
                       "low relative to the recent personal baseline")
    elif acwr_today > 1.3:
        session, mult = "light technique session", 0.6
        reasons.append(f"recent-to-baseline load ratio {acwr_today:.2f} is "
                       "materially elevated; the project heuristic avoids "
                       "adding more volume")
    else:
        target = deficit.index[0] if len(deficit) else "chest"
        session = f"{target} focus"
        if readiness > 0.5 and acwr_today < 1.1:
            mult = 1.1
            reasons.append(f"readiness heuristic {readiness:+.1f} sd and "
                           f"load ratio {acwr_today:.2f} are compatible with "
                           "the project's higher-volume branch")
        else:
            mult = 1.0
            reasons.append(f"readiness heuristic {readiness:+.1f} sd and "
                           f"load ratio {acwr_today:.2f} select the neutral "
                           "volume branch")
        if len(deficit):
            reasons.append(f"{target} is at {deficit.iloc[0]:.0%} of its "
                           "preceding-28-day weekly average, the largest gap")
    return {
        "session": session,
        "volume_multiplier": mult,
        "reasons": reasons,
        "status": "illustrative planning heuristic; not injury or medical advice",
    }
