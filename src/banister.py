"""Banister-style fitness-fatigue impulse-response model.

The implementation deliberately keeps the slower fitness time constant above
the faster fatigue time constant.  Even with that physical ordering, parameter
identifiability must be checked rather than inferred from a visually good fit.
"""
import numpy as np
from scipy.optimize import least_squares

TAU_FAST_MIN = 2.0
TAU_FAST_MAX = 30.0
TAU_SLOW_MAX = 80.0


def impulse_response(load: np.ndarray, tau: float) -> np.ndarray:
    n = len(load)
    out = np.zeros(n)
    decay = np.exp(-1.0 / tau)
    for t in range(1, n):
        out[t] = out[t - 1] * decay + load[t - 1]
    return out


def form_curve(load, p0, k1, k2, tau1, tau2):
    return (p0 + k1 * impulse_response(load, tau1)
            - k2 * impulse_response(load, tau2))


def _pack(p0: float, k1: float, k2: float,
          tau1: float, tau2: float) -> np.ndarray:
    q = (tau1 - tau2) / (TAU_SLOW_MAX - tau2)
    return np.array([p0, k1, k2, tau2, np.clip(q, 1e-3, 0.999)])


def _unpack(theta: np.ndarray) -> dict:
    p0, k1, k2, tau_fast, q = theta
    tau_slow = tau_fast + q * (TAU_SLOW_MAX - tau_fast)
    return {"p0": p0, "k1": k1, "k2": k2,
            "tau1": tau_slow, "tau2": tau_fast}


def fit_banister(load: np.ndarray, observed_idx: np.ndarray,
                 observed_form: np.ndarray, n_starts: int = 8,
                 random_state: int = 0):
    """Multi-start least-squares fit with tau1 > tau2 enforced.

    Multi-start fitting reduces dependence on one arbitrary initial guess; it
    does *not* solve structural/practical identifiability and the returned
    parameters should therefore be interpreted with diagnostics.
    """
    load = np.asarray(load, dtype=float)
    observed_idx = np.asarray(observed_idx, dtype=int)
    observed_form = np.asarray(observed_form, dtype=float)
    if len(observed_idx) != len(observed_form) or len(observed_idx) < 5:
        raise ValueError("Banister fit requires at least five aligned observations")
    if observed_idx.min() < 0 or observed_idx.max() >= len(load):
        raise ValueError("Observed indices fall outside the load series")
    scale = float(np.mean(observed_form))
    upper_p0 = max(2 * scale, scale + 1.0)

    def residuals(theta):
        params = _unpack(theta)
        return form_curve(load, **params)[observed_idx] - observed_form

    lower = np.array([0.0, 0.0, 0.0, TAU_FAST_MIN, 1e-3])
    upper = np.array([upper_p0, 20.0, 20.0, TAU_FAST_MAX, 0.999])
    starts = [_pack(scale, 0.05, 0.08, 30.0, 8.0)]
    rng = np.random.default_rng(random_state)
    for _ in range(max(0, n_starts - 1)):
        tau_fast = rng.uniform(4.0, 18.0)
        tau_slow = rng.uniform(max(tau_fast + 3.0, 20.0), 70.0)
        starts.append(_pack(
            np.clip(rng.normal(scale, max(abs(scale) * 0.08, 1.0)), 0, upper_p0),
            10 ** rng.uniform(-2, 0.8),
            10 ** rng.uniform(-2, 0.8),
            tau_slow, tau_fast,
        ))

    solutions = [least_squares(residuals, x0, bounds=(lower, upper))
                 for x0 in starts]
    sol = min(solutions, key=lambda x: x.cost)
    return _unpack(sol.x), sol


def taper_days(load: np.ndarray, params: dict, horizon: int = 21) -> int:
    """Rest days after the final load until predicted form peaks.

    Day 1 is the first day after the final training impulse.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one day")
    extended = np.concatenate([np.asarray(load, dtype=float), np.zeros(horizon)])
    curve = form_curve(extended, **params)
    tail = curve[len(load):]
    return int(np.argmax(tail) + 1)
