"""Known-truth audits for the training/recovery project.

These experiments are deliberately separate from the personal-data pathway.
They test whether the statistical/modeling machinery recovers quantities that
are known because the data-generating process was constructed by us.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from banister import fit_banister, form_curve
from predict import RECOVERY_ONLY, build_table, evaluate
from recovery import add_recovery_features, lagged_effect
from synthetic import make_athlete
from training import acwr, daily_load

ROOT = Path(__file__).resolve().parents[1]


def prediction_benchmark(n_worlds: int = 100) -> tuple[dict, pd.DataFrame]:
    rows = []
    for seed in range(n_worlds):
        workouts, metrics, _ = make_athlete(seed=1000 + seed)
        days = pd.date_range(metrics.date.min(), metrics.date.max(), freq="D")
        load = daily_load(workouts, days, metric="effective_tonnage")
        feats = add_recovery_features(metrics)
        ratio = acwr(load)
        table = build_table(feats, load, ratio)
        full = evaluate(table)
        recovery_only = evaluate(table, features=RECOVERY_ONLY)
        lag1 = lagged_effect(load, feats.hrv_z)[1]
        rows.append({
            "seed": seed,
            "improvement_vs_persistence_pct": full["improvement_pct"],
            "full_mae": full["mae_model"],
            "persistence_mae": full["mae_persistence"],
            "recovery_only_mae": recovery_only["mae_model"],
            "lag1_load_hrv": lag1,
            "load_coefficient": full["mean_standardised_coefficients"]["load_z"],
        })
    df = pd.DataFrame(rows)
    summary = {
        "worlds": int(n_worlds),
        "median_improvement_vs_persistence_pct": round(float(
            df.improvement_vs_persistence_pct.median()), 2),
        "p05_p95_improvement_pct": [round(float(x), 2) for x in
            df.improvement_vs_persistence_pct.quantile([0.05, 0.95])],
        "fraction_beating_persistence": round(float(
            (df.full_mae < df.persistence_mae).mean()), 3),
        "fraction_beating_recovery_only": round(float(
            (df.full_mae < df.recovery_only_mae).mean()), 3),
        "median_mae_full": round(float(df.full_mae.median()), 4),
        "median_mae_recovery_only": round(float(df.recovery_only_mae.median()), 4),
        "median_mae_persistence": round(float(df.persistence_mae.median()), 4),
        "median_lag1_load_hrv_correlation": round(float(df.lag1_load_hrv.median()), 3),
        "fraction_negative_load_coefficient": round(float(
            (df.load_coefficient < 0).mean()), 3),
        "interpretation": (
            "Controlled synthetic validation only; it establishes that the pipeline "
            "can recover the coupling deliberately planted in this generator, not "
            "that HRV is predictably driven by training load in a real athlete."
        ),
    }
    return summary, df


def _structured_load(seed: int, n_days: int = 140) -> np.ndarray:
    rng = np.random.default_rng(seed)
    load = np.zeros(n_days)
    for t in range(n_days):
        if t % 7 in {0, 1, 3, 4} and rng.random() > 0.08:
            dose = rng.normal(6.5, 1.0)
            if rng.random() < 0.25:
                dose *= 1.3
            load[t] = np.clip(dose, 0, 10)
    return load


def banister_identifiability_benchmark(n_worlds: int = 100) -> tuple[dict, pd.DataFrame]:
    true = {"p0": 100.0, "k1": 1.0, "k2": 1.7,
            "tau1": 35.0, "tau2": 9.0}
    scenarios = {
        "dense_clean": {"step": 2, "noise_sd": 0.0},
        "weekly_noisy": {"step": 7, "noise_sd": 3.0},
    }
    rows = []
    for world in range(n_worlds):
        load = _structured_load(5000 + world)
        truth = form_curve(load, **true)
        for name, cfg in scenarios.items():
            rng = np.random.default_rng(9000 + 1000 * world + cfg["step"])
            idx = np.arange(0, len(load), cfg["step"])
            obs = truth[idx] + rng.normal(0, cfg["noise_sd"], len(idx))
            params, sol = fit_banister(load, idx, obs, n_starts=2,
                                       random_state=world)
            fitted = form_curve(load, **params)
            rows.append({
                "world": world,
                "scenario": name,
                **{k: float(params[k]) for k in ["p0", "k1", "k2", "tau1", "tau2"]},
                "tau1_rel_error": abs(params["tau1"] - true["tau1"]) / true["tau1"],
                "tau2_rel_error": abs(params["tau2"] - true["tau2"]) / true["tau2"],
                "observed_rmse": float(np.sqrt(np.mean((fitted[idx] - obs) ** 2))),
                "full_truth_rmse": float(np.sqrt(np.mean((fitted - truth) ** 2))),
                "cost": float(sol.cost),
            })
    df = pd.DataFrame(rows)
    summary = {"worlds_per_scenario": int(n_worlds), "truth": true, "scenarios": {}}
    for name in scenarios:
        d = df[df.scenario == name]
        summary["scenarios"][name] = {
            "observation_spacing_days": scenarios[name]["step"],
            "observation_noise_sd": scenarios[name]["noise_sd"],
            "tau1_median": round(float(d.tau1.median()), 3),
            "tau1_p05_p95": [round(float(x), 3) for x in d.tau1.quantile([0.05, 0.95])],
            "tau2_median": round(float(d.tau2.median()), 3),
            "tau2_p05_p95": [round(float(x), 3) for x in d.tau2.quantile([0.05, 0.95])],
            "median_tau1_relative_error_pct": round(float(100 * d.tau1_rel_error.median()), 2),
            "median_tau2_relative_error_pct": round(float(100 * d.tau2_rel_error.median()), 2),
            "median_observed_rmse": round(float(d.observed_rmse.median()), 3),
            "median_full_truth_rmse": round(float(d.full_truth_rmse.median()), 3),
        }
    summary["interpretation"] = (
        "Dense noiseless data are a code-recovery check. Weekly noisy observations "
        "stress practical identifiability; broad parameter ranges can coexist with "
        "reasonable fits, so fitted time constants are not physiological facts."
    )
    return summary, df


def write_figure(pred_df: pd.DataFrame, ban_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].hist(pred_df.improvement_vs_persistence_pct, bins=18)
    axes[0].axvline(pred_df.improvement_vs_persistence_pct.median(), ls="--")
    axes[0].set(xlabel="MAE improvement vs persistence (%)",
                ylabel="Synthetic worlds", title="Next-day HRV benchmark")

    for scenario, marker in [("dense_clean", "o"), ("weekly_noisy", "x")]:
        d = ban_df[ban_df.scenario == scenario]
        axes[1].scatter(d.tau2, d.tau1, s=18, alpha=0.55,
                        marker=marker, label=scenario)
    axes[1].axvline(9.0, ls="--")
    axes[1].axhline(35.0, ls="--")
    axes[1].set(xlabel="Fitted fast tau (days)", ylabel="Fitted slow tau (days)",
                title="Banister parameter recovery")
    axes[1].legend(fontsize=8)

    noisy = ban_df[ban_df.scenario == "weekly_noisy"]
    axes[2].scatter(noisy.observed_rmse, noisy.tau2_rel_error * 100,
                    s=18, alpha=0.55)
    axes[2].set(xlabel="Observed-point RMSE", ylabel="Fast-tau error (%)",
                title="Good fit can hide parameter error")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=100)
    args = ap.parse_args()
    pred, pred_df = prediction_benchmark(args.worlds)
    ban, ban_df = banister_identifiability_benchmark(args.worlds)
    payload = {"prediction_known_truth": pred,
               "banister_identifiability": ban}
    out = ROOT / "reports" / "audit_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    pred_df.to_csv(ROOT / "reports" / "prediction_worlds.csv", index=False)
    ban_df.to_csv(ROOT / "reports" / "banister_worlds.csv", index=False)
    write_figure(pred_df, ban_df,
                 ROOT / "reports" / "figures" / "known_truth_audits.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
