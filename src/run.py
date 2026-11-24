"""Run the descriptive training/recovery pipeline and write figures/summary."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from banister import fit_banister, form_curve, taper_days
from ingest import load_apple_health, load_hevy
from predict import build_table, evaluate
from recommend import recommend, volume_deficit
from recovery import add_recovery_features, lagged_effect
from synthetic import make_athlete
from training import acwr, daily_load, e1rm_series, personal_records, weekly_volume

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hevy", default=None)
    ap.add_argument("--apple", default=None)
    ap.add_argument("--key-lift", default="Bench Press (Barbell)")
    ap.add_argument("--load-metric", choices=["tonnage", "effective_tonnage"],
                    default="effective_tonnage",
                    help="RPE-weighted heuristic by default; missing RPE maps to 8")
    args = ap.parse_args()

    if args.synthetic:
        workouts, metrics, _ = make_athlete()
        label = "synthetic athlete"
    elif args.hevy and args.apple:
        workouts, metrics = load_hevy(args.hevy), load_apple_health(args.apple)
        label = "personal data (local only)"
    else:
        ap.error("choose --synthetic or provide both --hevy and --apple")

    days = pd.date_range(metrics.date.min(), metrics.date.max(), freq="D")
    load = daily_load(workouts, days, metric=args.load_metric)
    ratio = acwr(load)
    feats = add_recovery_features(metrics)
    FIGURES.mkdir(parents=True, exist_ok=True)

    wv = weekly_volume(workouts) / 1000
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for muscle in wv.columns:
        ax.plot(wv.index, wv[muscle], lw=1.6, label=muscle)
    ax.set(ylabel="Weekly tonnage (t)", title=f"Weekly volume by muscle ({label})")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "weekly_volume.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    for lift in workouts.exercise_title.value_counts().head(3).index:
        s = e1rm_series(workouts, lift)
        ax2.plot(s.index, s, lw=1.5, label=lift)
        pr = personal_records(s)
        ax2.plot(pr.index, pr, "o", ms=4)
    ax2.set(ylabel="Estimated 1RM (kg, Epley)",
            title="Strength trajectories; strict estimated-PR improvements marked")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "e1rm_progress.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(9, 4.2))
    ax3.axhline(1.0, ls="--", lw=1.0, label="recent = longer-horizon EWMA")
    ax3.plot(ratio.index, ratio, lw=1.6)
    ax3.set(ylabel="Recent:longer-horizon workload ratio",
            title="Descriptive load-balance ratio (not an injury-risk score)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIGURES / "acwr.png", dpi=150)

    fig4, ax4 = plt.subplots(figsize=(9.5, 4.6))
    ax4.plot(feats.index, feats.hrv_z, lw=1.4,
             label="HRV z vs prior 28-day personal baseline")
    ax4.axhline(0, lw=0.8)
    ax4b = ax4.twinx()
    ax4b.plot(load.index, load / 1000, lw=0.8, alpha=0.6,
              label=args.load_metric.replace("_", " "))
    ax4b.set_ylabel(f"{args.load_metric.replace('_', ' ').title()} (t)")
    ax4.set(ylabel="HRV z", title="Recovery signal against training load")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(FIGURES / "hrv_vs_load.png", dpi=150)

    e1 = e1rm_series(workouts, args.key_lift)
    if len(e1) < 5:
        raise ValueError(f"Need at least five {args.key_lift} observations for Banister fit")
    idx = np.array([days.get_loc(d) for d in e1.index])
    max_load = float(load.max())
    if max_load <= 0:
        raise ValueError("Cannot fit Banister model without positive training load")
    norm_load = (load / max_load * 10).to_numpy()
    params, fit = fit_banister(norm_load, idx, e1.to_numpy())
    curve = form_curve(norm_load, **params)
    fig5, ax5 = plt.subplots(figsize=(9.5, 4.8))
    ax5.plot(days, curve, lw=1.8, label="fitted form")
    ax5.plot(e1.index, e1, "o", ms=4,
             label=f"observed e1RM ({args.key_lift})")
    ax5.set(ylabel="Form (kg e1RM)",
            title="Banister-style fit: descriptive fit, not physiological truth")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(FIGURES / "banister.png", dpi=150)
    plt.close("all")

    table = build_table(feats, load, ratio)
    pred = evaluate(table)
    lags = lagged_effect(load, feats.hrv_z)

    asof = days[-1]
    readiness = float(feats.readiness.iloc[-1])
    rec = recommend(readiness, float(ratio.iloc[-1]), volume_deficit(workouts, asof))

    summary = {
        "data": label,
        "load_metric": args.load_metric,
        "sets": int(len(workouts)),
        "days": int(len(days)),
        "recovery_baseline": "prior 28 days only (causal)",
        "load_ratio_interpretation": "descriptive only; no injury-risk threshold",
        "banister_fit": {
            "params": {k: round(float(v), 3) for k, v in params.items()},
            "ordered_time_constants": bool(params["tau1"] > params["tau2"]),
            "observed_rmse": round(float(np.sqrt(2 * fit.cost / len(e1))), 3),
            "gain_at_upper_bound": bool(params["k1"] > 19.9 or params["k2"] > 19.9),
            "interpretation": "fit parameters require identifiability diagnostics",
        },
        "taper_days_to_peak_model_form": taper_days(norm_load, params),
        "load_to_hrv_lag_correlations": {k: round(v, 3) for k, v in lags.items()},
        "next_day_hrv_prediction": pred,
        "todays_planning_heuristic": rec,
    }
    out = ROOT / "reports" / "summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures -> {FIGURES}")


if __name__ == "__main__":
    main()
