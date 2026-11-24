"""A coupled synthetic athlete: eight months of workouts and morning
metrics with a known ground truth, so the whole pipeline runs and is
testable without anyone's real health data.

The couplings are the ones the analysis should find: hard training days
depress the next morning's HRV and nudge resting heart rate up, short
sleep does the same, and strength follows a Banister-style process driven by a known planned
training impulse.  The realised set tonnage is intentionally not treated as
identical to that latent impulse, so the end-to-end fit is an analysis example
rather than a parameter-recovery benchmark.
"""
import numpy as np
import pandas as pd

SPLIT = {0: "Push Day", 1: "Pull Day", 3: "Leg Day", 4: "Upper Day"}
LIFTS = {
    "Push Day": [("Bench Press (Barbell)", "chest", 75),
                 ("Overhead Press (Barbell)", "shoulders", 45)],
    "Pull Day": [("Deadlift (Barbell)", "posterior", 130),
                 ("Lat Pulldown (Cable)", "back", 60)],
    "Leg Day": [("Squat (Barbell)", "quads", 100),
                ("Leg Press", "quads", 170)],
    "Upper Day": [("Bench Press (Barbell)", "chest", 75),
                  ("Row (Barbell)", "back", 70),
                  ("Bicep Curl (Dumbbell)", "arms", 14)],
}


def banister_form(load: np.ndarray, p0=100.0, k1=0.06, k2=0.10,
                  tau1=35.0, tau2=9.0) -> np.ndarray:
    n = len(load)
    fitness = np.zeros(n)
    fatigue = np.zeros(n)
    for t in range(1, n):
        fitness[t] = fitness[t - 1] * np.exp(-1 / tau1) + load[t - 1]
        fatigue[t] = fatigue[t - 1] * np.exp(-1 / tau2) + load[t - 1]
    return p0 + k1 * fitness - k2 * fatigue


def make_athlete(n_days: int = 240, seed: int = 11):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-06")
    dates = pd.date_range(start, periods=n_days, freq="D")

    # Pass 1: the training plan (sessions, hardness, base loads) fixes the
    # impulse pattern; pass 2 lets realised weights follow the Banister
    # form that pattern produces, so strength genuinely carries
    # fitness-fatigue dynamics the fitter can recover.
    plan = []
    plan_load = np.zeros(n_days)
    for i, d in enumerate(dates):
        session = SPLIT.get(d.weekday())
        if session is None or rng.random() < 0.08:      # occasional skip
            continue
        hard = rng.random() < 0.3
        plan.append((i, d, session, hard))
        for _, _, base in LIFTS[session]:
            plan_load[i] += base * 8 * (4 if hard else 3)

    impulse = plan_load / plan_load.max() * 10
    true_params = {"p0": 1.0, "k1": 0.010, "k2": 0.018,
                   "tau1": 35.0, "tau2": 9.0}
    form_rel = banister_form(impulse, **true_params)

    sets, daily_load = [], np.zeros(n_days)
    for i, d, session, hard in plan:
        for exercise, muscle, base in LIFTS[session]:
            weight = base * form_rel[i] * (1.05 if hard else 1.0)
            weight = round(weight * (1 + rng.normal(0, 0.015)) / 2.5) * 2.5
            n_sets = 4 if hard else 3
            for s in range(n_sets):
                reps = int(np.clip(rng.normal(8 - 1.5 * hard, 1.2), 3, 12))
                rpe = float(np.clip(rng.normal(8.6 if hard else 7.4, 0.5),
                                    6.0, 10.0))
                sets.append({"date": d, "title": session,
                             "exercise_title": exercise, "muscle": muscle,
                             "set_type": "normal", "working": True,
                             "weight_kg": weight, "reps": reps,
                             "rpe": round(rpe * 2) / 2,
                             "tonnage": weight * reps,
                             "effective_tonnage": weight * reps * rpe / 8.0})
                daily_load[i] += weight * reps

    load_z = (daily_load - daily_load.mean()) / daily_load.std()
    lag_load = np.concatenate([[0.0], load_z[:-1]])
    sleep = np.clip(rng.normal(7.2, 0.9, n_days)
                    - 0.15 * lag_load, 4.5, 9.5)
    sleep_z = (sleep - sleep.mean()) / sleep.std()
    hrv = np.zeros(n_days)
    hrv[0] = 55
    for t in range(1, n_days):
        hrv[t] = (55 + 0.55 * (hrv[t - 1] - 55)
                  - 3.2 * max(load_z[t - 1], 0)      # hard day, low HRV
                  + 1.6 * sleep_z[t]
                  + rng.normal(0, 2.4))
    rhr = 58 - 0.28 * (hrv - 55) + rng.normal(0, 1.1, n_days)

    workouts = pd.DataFrame(sets)
    metrics = pd.DataFrame({"date": dates, "hrv": hrv, "rhr": rhr,
                            "sleep_hours": sleep})
    truth = {"daily_load": daily_load, "params": true_params,
             "form_rel": form_rel}
    return workouts, metrics, truth
