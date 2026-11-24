"""Generate a descriptive weekly training report.

The recent-to-baseline load ratio and readiness score are project heuristics;
they are not injury-risk, medical or validated coaching models.
"""
import argparse
from pathlib import Path

import pandas as pd

from ingest import load_hevy
from synthetic import make_athlete
from training import acwr, daily_load, e1rm_series

ROOT = Path(__file__).resolve().parents[1]
KEY_LIFTS = ["Bench Press (Barbell)", "Squat (Barbell)",
             "Deadlift (Barbell)"]


def acwr_flag(ratio: float) -> str:
    """Descriptive language only; cut points are display heuristics."""
    if ratio != ratio:
        return "not enough history yet"
    if ratio < 0.8:
        return f"{ratio:.2f}, recent load below the longer-horizon baseline"
    if ratio <= 1.3:
        return f"{ratio:.2f}, recent and longer-horizon load broadly similar"
    if ratio <= 1.5:
        return f"{ratio:.2f}, recent load above the longer-horizon baseline"
    return f"{ratio:.2f}, recent load much higher than the longer-horizon baseline"


def build_report(workouts: pd.DataFrame) -> str:
    days = pd.date_range(workouts.date.min(), workouts.date.max(), freq="D")
    load = daily_load(workouts, days, metric="effective_tonnage")
    ratio = float(acwr(load).iloc[-1])

    last7 = workouts[workouts.date > days[-1] - pd.Timedelta(days=7)]
    prev28 = workouts[(workouts.date <= days[-1] - pd.Timedelta(days=7))
                      & (workouts.date > days[-1] - pd.Timedelta(days=35))]
    by_muscle = last7[last7.working].groupby("muscle").tonnage.sum()
    baseline = prev28[prev28.working].groupby("muscle").tonnage.sum() / 4

    lines = [f"# Weekly training report, week ending {days[-1].date()}", ""]
    lines.append(f"**Load balance.** Recent-to-baseline ratio {acwr_flag(ratio)}.")
    lines.append("_Descriptive planning heuristic only; not an injury-risk threshold._")
    lines.append("")
    lines.append("**Volume by muscle, last 7 days vs preceding 28-day weekly average:**")
    lines.append("")
    for muscle in sorted(set(by_muscle.index) | set(baseline.index)):
        now = by_muscle.get(muscle, 0.0)
        base = baseline.get(muscle, 0.0)
        move = "up" if now > base * 1.1 else \
            "down" if now < base * 0.9 else "steady"
        lines.append(f"- {muscle}: {now:,.0f} kg ({move} vs {base:,.0f} kg average)")
    lines.append("")
    lines.append("**Key lift e1RM estimates:**")
    lines.append("")
    for lift in KEY_LIFTS:
        series = e1rm_series(workouts, lift)
        if len(series):
            lines.append(f"- {lift}: {series.iloc[-1]:.1f} kg (peak {series.max():.1f})")
    lines.append("")
    quiet = baseline.drop(index=by_muscle.index, errors="ignore")
    if ratio > 1.3:
        rec = ("Planning flag: recent load is elevated relative to baseline; "
               "review recovery and programme intent before adding volume.")
    elif len(quiet):
        rec = f"Planning flag: {quiet.idxmax()} received no work this week."
    else:
        rec = "Planning flag: recent muscle-group volume is broadly balanced."
    lines.append(f"**Heuristic note.** {rec}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hevy")
    ap.add_argument("--out", default=ROOT / "reports" / "weekly_report.md")
    args = ap.parse_args()
    if args.synthetic:
        workouts, _, _ = make_athlete()
    elif args.hevy:
        workouts = load_hevy(args.hevy)
    else:
        ap.error("choose --synthetic or provide --hevy")
    report = build_report(workouts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report[:500])
    print(f"...\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
