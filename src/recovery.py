"""Recovery features built from morning metrics.

Rolling baselines are causal: today's observation is compared with the
*previous* 28 days rather than helping to define its own baseline.  That
keeps the same feature usable in retrospective analysis and prospective
prediction without future leakage or self-normalisation.
"""
import numpy as np
import pandas as pd


def rolling_z(s: pd.Series, window: int = 28, min_periods: int = 10) -> pd.Series:
    """Causal rolling z-score against prior observations only."""
    history = s.shift(1)
    mean = history.rolling(window, min_periods=min_periods).mean()
    sd = history.rolling(window, min_periods=min_periods).std()
    sd = sd.where(sd > 0)
    return ((s - mean) / sd).replace([np.inf, -np.inf], np.nan)


def add_recovery_features(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy().sort_values("date").set_index("date")
    m["hrv_z"] = rolling_z(m["hrv"])
    m["rhr_z"] = rolling_z(m["rhr"])
    m["sleep_z"] = rolling_z(m["sleep_hours"])
    # Transparent project heuristic: higher HRV, lower resting HR and more
    # sleep raise the score.  It is not a validated medical/readiness index.
    m["readiness"] = m["hrv_z"] - 0.5 * m["rhr_z"] + 0.5 * m["sleep_z"]
    return m


def lagged_effect(load: pd.Series, hrv_z: pd.Series,
                  max_lag: int = 3) -> dict[int, float]:
    """Correlation between load on day t and HRV z on day t+k.

    This is descriptive association, not a causal effect estimate.
    """
    sd = load.std()
    if not np.isfinite(sd) or sd == 0:
        return {k: float("nan") for k in range(max_lag + 1)}
    load_z = (load - load.mean()) / sd
    out = {}
    for k in range(0, max_lag + 1):
        out[k] = float(load_z.corr(hrv_z.shift(-k)))
    return out
