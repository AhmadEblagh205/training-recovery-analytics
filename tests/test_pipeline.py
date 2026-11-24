import numpy as np
import pandas as pd
import pytest

from banister import fit_banister, form_curve, taper_days
from ingest import load_apple_health, load_hevy, muscle_of
from predict import build_table, evaluate
from recommend import recommend, volume_deficit
from recovery import add_recovery_features, lagged_effect, rolling_z
from synthetic import make_athlete
from training import (acwr, daily_load, epley_1rm, personal_records,
                      weekly_volume)


def test_epley_known_values():
    assert epley_1rm(100, 1) == 100 * (1 + 1 / 30)
    assert np.isclose(epley_1rm(80, 8), 80 * (1 + 8 / 30))


def test_hevy_loader_on_sample():
    df = load_hevy("data/sample/hevy_sample.csv")
    assert set(df.muscle) >= {"chest", "quads", "posterior", "back"}
    assert df.loc[df.set_type == "warmup", "tonnage"].eq(0).all()
    assert df.loc[(df.exercise_title == "Squat (Barbell)")
                  & df.working, "tonnage"].sum() == 110 * 6 + 110 * 5


def test_hevy_loader_fails_on_bad_schema_and_values(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"start_time": ["2025-01-01"], "exercise_title": ["Squat"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_hevy(bad)

    src = pd.read_csv("data/sample/hevy_sample.csv")
    src.loc[0, "weight_kg"] = -1
    src.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="cannot be negative"):
        load_hevy(bad)


def test_muscle_mapping_falls_back():
    assert muscle_of("Bench Press (Barbell)") == "chest"
    assert muscle_of("Obscure Machine Thing") == "other"


def test_apple_health_stream_parser(tmp_path):
    xml = tmp_path / "export.xml"
    xml.write_text("""<?xml version="1.0"?>
<HealthData>
 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
  startDate="2025-03-01 07:00:00 +0000" endDate="2025-03-01 07:01:00 +0000"
  value="61.5"/>
 <Record type="HKQuantityTypeIdentifierRestingHeartRate"
  startDate="2025-03-01 09:00:00 +0000" endDate="2025-03-01 09:00:00 +0000"
  value="55"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
  value="HKCategoryValueSleepAnalysisAsleepCore"
  startDate="2025-02-28 23:30:00 +0000" endDate="2025-03-01 06:30:00 +0000"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
  value="HKCategoryValueSleepAnalysisInBed"
  startDate="2025-02-28 23:00:00 +0000" endDate="2025-03-01 07:00:00 +0000"/>
</HealthData>""")
    df = load_apple_health(xml)
    row = df.set_index("date").loc[pd.Timestamp("2025-03-01")]
    assert row.hrv == 61.5 and row.rhr == 55
    assert np.isclose(row.sleep_hours, 7.0)


def test_apple_sleep_merges_overlaps_and_respects_offsets(tmp_path):
    xml = tmp_path / "export.xml"
    xml.write_text("""<HealthData>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
  value="HKCategoryValueSleepAnalysisAsleepCore"
  startDate="2025-03-29 23:00:00 +0000" endDate="2025-03-30 03:00:00 +0100"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
  value="HKCategoryValueSleepAnalysisAsleepREM"
  startDate="2025-03-30 01:00:00 +0000" endDate="2025-03-30 03:00:00 +0100"/>
</HealthData>""")
    row = load_apple_health(xml).iloc[0]
    # 23:00 UTC -> 02:00 UTC is three actual hours; overlapping REM must not double count.
    assert np.isclose(row.sleep_hours, 3.0)
    assert row.date == pd.Timestamp("2025-03-30")


def test_rolling_z_is_causal_and_future_safe():
    idx = pd.date_range("2025-01-01", periods=40)
    s = pd.Series(np.arange(40.0), index=idx)
    z = rolling_z(s, window=10, min_periods=5)
    changed = s.copy()
    changed.iloc[30:] += 1000
    z_changed = rolling_z(changed, window=10, min_periods=5)
    pd.testing.assert_series_equal(z.iloc[:30], z_changed.iloc[:30])
    prior = s.iloc[0:10]
    expected = (s.iloc[10] - prior.mean()) / prior.std()
    assert np.isclose(z.iloc[10], expected)


def test_acwr_is_near_one_for_steady_training():
    load = pd.Series(1000.0, index=pd.date_range("2025-01-01", periods=90))
    r = acwr(load)
    assert abs(r.iloc[-1] - 1.0) < 0.01


def test_banister_recovers_ordered_time_constants_on_clean_data():
    rng = np.random.default_rng(0)
    load = rng.uniform(0, 10, 200) * (rng.random(200) < 0.5)
    true = dict(p0=100.0, k1=1.2, k2=2.0, tau1=40.0, tau2=7.0)
    form = form_curve(load, **true)
    idx = np.arange(0, 200, 2)
    params, _ = fit_banister(load, idx, form[idx])
    assert params["tau1"] > params["tau2"]
    assert abs(params["tau1"] - true["tau1"]) / true["tau1"] < 0.05
    assert abs(params["tau2"] - true["tau2"]) / true["tau2"] < 0.05
    assert 1 <= taper_days(load, params) <= 21


def test_banister_preserves_ordering_under_noise():
    rng = np.random.default_rng(4)
    load = rng.uniform(0, 10, 120) * (rng.random(120) < 0.5)
    true = dict(p0=100.0, k1=1.0, k2=1.7, tau1=35.0, tau2=9.0)
    form = form_curve(load, **true)
    idx = np.arange(0, 120, 7)
    obs = form[idx] + rng.normal(0, 3, len(idx))
    params, _ = fit_banister(load, idx, obs)
    assert params["tau1"] > params["tau2"]


def test_synthetic_couplings_are_learnable_without_cv_overlap():
    workouts, metrics, _ = make_athlete()
    days = pd.date_range(metrics.date.min(), metrics.date.max(), freq="D")
    load = daily_load(workouts, days)
    feats = add_recovery_features(metrics)
    lags = lagged_effect(load, feats.hrv_z)
    assert lags[1] < -0.2
    table = build_table(feats, load, acwr(load))
    res = evaluate(table)
    assert res["mae_model"] < res["mae_persistence"]
    assert res["mean_standardised_coefficients"]["load_z"] < 0
    assert res["cv_gap_days"] == 1


def test_prediction_table_is_future_safe():
    workouts, metrics, _ = make_athlete(n_days=100, seed=3)
    days = pd.date_range(metrics.date.min(), metrics.date.max(), freq="D")
    load = daily_load(workouts, days)
    feats = add_recovery_features(metrics)
    t1 = build_table(feats, load, acwr(load))
    changed = load.copy()
    changed.iloc[80:] *= 20
    t2 = build_table(feats, changed, acwr(changed))
    common = t1.index[t1.index < days[80]]
    pd.testing.assert_series_equal(t1.loc[common, "load_z"], t2.loc[common, "load_z"])


def test_recommender_branches_and_caveat():
    empty = pd.Series(dtype=float)
    low = recommend(-1.5, 1.0, empty)
    assert low["session"] == "rest or mobility"
    assert "not injury or medical advice" in low["status"]
    assert recommend(0.0, 1.5, empty)["volume_multiplier"] == 0.6
    push = recommend(0.8, 0.95, pd.Series({"back": 0.4, "chest": 1.1}))
    assert push["session"] == "back focus" and push["volume_multiplier"] == 1.1


def test_volume_deficit_uses_preceding_baseline_not_overlapping_window():
    rows = []
    asof = pd.Timestamp("2025-02-05")
    for d in pd.date_range(asof - pd.Timedelta(days=35), asof, freq="D"):
        if d <= asof - pd.Timedelta(days=7):
            rows.append({"date": d, "working": True, "muscle": "back", "tonnage": 100.0})
        else:
            rows.append({"date": d, "working": True, "muscle": "back", "tonnage": 200.0})
    w = pd.DataFrame(rows)
    ratio = volume_deficit(w, asof)
    # Seven recent days at 200 vs 28 prior days at 100 -> 1400 / 700 = 2.
    assert np.isclose(ratio["back"], 2.0)


def test_weekly_volume_shape():
    workouts, _, _ = make_athlete(n_days=60)
    wv = weekly_volume(workouts)
    assert (wv.to_numpy() >= 0).all()
    assert "quads" in wv.columns


def test_personal_records_require_strict_improvement():
    s = pd.Series([100.0, 100.0, 102.0, 101.0, 102.0, 103.0],
                  index=pd.date_range("2025-01-01", periods=6))
    prs = personal_records(s)
    assert prs.tolist() == [100.0, 102.0, 103.0]


def test_rpe_weighting_orders_and_falls_back():
    df = load_hevy("data/sample/hevy_sample.csv")
    bench = df[(df.exercise_title == "Bench Press (Barbell)") & df.working]
    e = bench.set_index("rpe")["effective_tonnage"]
    assert e.loc[9.0] > 640 * 7 / 8 * 0.99 and e.loc[9.0] > e.loc[8.0] * 0.85
    ohp = df[df.exercise_title == "Lateral Raise (Dumbbell)"].iloc[0]
    assert np.isclose(ohp.effective_tonnage, ohp.tonnage)


def test_weekly_report_is_descriptive_not_sweet_spot_claim():
    from report import acwr_flag, build_report
    workouts, _, _ = make_athlete(n_days=120, seed=4)
    report = build_report(workouts)
    assert report.startswith("# Weekly training report")
    assert "Load balance" in report and "Heuristic note" in report
    assert "sweet spot" not in report.lower()
    assert "recent and longer-horizon" in acwr_flag(1.0)
    assert "much higher" in acwr_flag(1.8)


def test_small_known_truth_audits_behave_as_expected():
    from audit import banister_identifiability_benchmark, prediction_benchmark
    pred, _ = prediction_benchmark(5)
    assert pred["fraction_beating_persistence"] >= 0.8
    ban, _ = banister_identifiability_benchmark(5)
    dense = ban["scenarios"]["dense_clean"]
    noisy = ban["scenarios"]["weekly_noisy"]
    assert dense["median_tau1_relative_error_pct"] < 1
    assert dense["median_tau2_relative_error_pct"] < 1
    assert (noisy["tau1_p05_p95"][1] - noisy["tau1_p05_p95"][0]) > 1
