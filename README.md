# Training-Recovery Modelling & Identifiability Laboratory

[![CI](https://github.com/AhmadEblagh205/training-recovery-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/training-recovery-analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I record every gym set in Hevy and my recovery signals in Apple Health, and I built this to see what those measurements can support. The repository has an optional private pipeline for my own exports, which stay outside git, and checked-in experiments on synthetic data. All the numbers below come from the synthetic data. It is an analytics and model-validation project. It is not an injury predictor, a medical device or a coaching system.

## What it does

- Reads Hevy set exports and Apple Health `export.xml` locally, and computes tonnage, an RPE-weighted tonnage, Epley estimated 1RM and weekly muscle-group volume.
- Compares short and long exponentially weighted load (the `acwr` function) as a descriptive signal, and builds recovery features against a personal baseline from the previous 28 days only.
- Predicts next-day HRV with a ridge model, using time-series splits, a one-day embargo and persistence as the baseline.
- Fits a Banister-style fitness-fatigue model and tests whether its time constants can be recovered.
- Writes a weekly markdown report and a planning suggestion that lists its reasons.

![Known-truth audits](reports/figures/known_truth_audits.png)

## Results

On the synthetic athlete, load on day t is most strongly related to HRV on day t+1, with a lag-1 correlation of about -0.40. That is recovery of a relationship I planted, not an estimate for a real athlete.

Next-day HRV prediction across 100 synthetic athletes:

- median MAE improvement over persistence of 28.25% (5th to 95th percentile 22.68 to 34.21%);
- the full model beat persistence and beat a recovery-only model in 100 of 100 worlds;
- median lag-1 load and HRV correlation of -0.424, with a negative load coefficient in 100 of 100 worlds.

The generator contains a load effect by construction, so this checks the code and the estimator. It does not show that a real person's HRV can be predicted this well.

Banister time constants, with a true slow constant of 35 days and a true fast constant of 9 days, across 100 worlds:

| Observations | Slow tau median (5th to 95th) | Fast tau median (5th to 95th) | Median relative error |
|---|---|---|---|
| Every 2 days, no noise | 35 | 9 | 0% |
| Every 7 days, noise SD 3 | 35.73 (20.48 to 48.48) | 8.31 (4.34 to 19.87) | 13.0% slow, 27.6% fast |

With sparse noisy data a model can fit well while its parameters stay uncertain. This links to my doctoral work on practical identifiability and informative experiment design. I do not report the end-to-end synthetic fit as parameter recovery, because set tonnage is only a proxy for the generator's training impulse, and in that run one gain reached its optimisation bound.

## How to run

```bash
pip install -r requirements.txt

# Checked-in synthetic pipeline
python src/run.py --synthetic
python src/report.py --synthetic

# Known-truth audits
python src/audit.py --worlds 100

# Local personal-data path; raw exports stay gitignored
python src/run.py --hevy data/raw/workouts.csv --apple data/raw/export.xml
python src/report.py --hevy data/raw/workouts.csv

pytest
```

The 19 tests pass. The command line stops if real-data mode is only partly specified, and does not fall back to synthetic data.

## Notes

The readiness score, the load ratio and the planning suggestion are heuristics. I make no claim that a load-ratio threshold prevents injury or that the fitted Banister constants are physiological constants. The pipeline details, formulas, caveats and references are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Licence

MIT, see [LICENSE](LICENSE).
