"""One-step recovery prediction with leakage-aware time-series evaluation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recovery import rolling_z

FEATURES = ["load_z", "sleep_z", "hrv_z", "acwr"]
RECOVERY_ONLY = ["sleep_z", "hrv_z"]


def build_table(metrics_feat: pd.DataFrame, load: pd.Series,
                acwr: pd.Series) -> pd.DataFrame:
    """Create a causal day-t -> day-(t+1) prediction table.

    Load is standardised against its own previous 28 days.  The target is
    tomorrow's HRV z-score; the final row therefore has no target.
    """
    t = metrics_feat.copy().sort_index()
    aligned_load = load.reindex(t.index)
    t["load_z"] = rolling_z(aligned_load, window=28, min_periods=10)
    t["acwr"] = acwr.reindex(t.index)
    t["target"] = t["hrv_z"].shift(-1)
    return t.dropna(subset=FEATURES + ["target"])


def evaluate(table: pd.DataFrame, splits: int = 5,
             features: list[str] | None = None) -> dict:
    """Expanding time-series CV against persistence.

    `gap=1` is deliberate: a training row's target is the next day's HRV,
    so without an embargo that same value would reappear as the first test
    row's current-HRV feature.  Scaling is fitted inside each training fold.
    """
    features = FEATURES if features is None else features
    X = table[features].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=float)
    persist = table["hrv_z"].to_numpy(dtype=float)
    maes_model, maes_persist, coefs = [], [], []
    splitter = TimeSeriesSplit(n_splits=splits, gap=1)
    for tr, te in splitter.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        maes_model.append(np.mean(np.abs(model.predict(X[te]) - y[te])))
        maes_persist.append(np.mean(np.abs(persist[te] - y[te])))
        coefs.append(model.named_steps["ridge"].coef_)
    mean_model = float(np.mean(maes_model))
    mean_persist = float(np.mean(maes_persist))
    return {
        "mae_model": round(mean_model, 4),
        "mae_persistence": round(mean_persist, 4),
        "improvement_pct": round(100 * (1 - mean_model / mean_persist), 1),
        "fold_mae_model": [round(float(x), 4) for x in maes_model],
        "fold_mae_persistence": [round(float(x), 4) for x in maes_persist],
        "mean_standardised_coefficients": {
            f: round(float(c), 4)
            for f, c in zip(features, np.mean(coefs, axis=0))
        },
        "n_obs": int(len(table)),
        "cv_gap_days": 1,
    }
