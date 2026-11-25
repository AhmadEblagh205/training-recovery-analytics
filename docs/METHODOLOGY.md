# Methodology and defensibility notes

This document records the assumptions that matter for interpreting the repository. The project is an analytics/model-validation demonstration, not a medical or injury-prevention system.

## 1. Data provenance

Two execution paths exist:

1. **Local personal-data path** — Hevy CSV + Apple Health XML. Raw exports live under `data/raw/` and are gitignored.
2. **Checked-in evidence path** — a coupled synthetic athlete and explicit known-truth audits. Recruiter-facing numerical claims should come from this path unless a future real-data study is separately documented.

Synthetic data support statements about whether the code recovers a planted mechanism. They do not establish that the same mechanism or effect size exists in a real athlete.

## 2. Training metrics

### Tonnage

For a working set:

\[
L_{set}=w\times r.
\]

Warm-up sets are retained in the tidy data but assigned zero tonnage for workload summaries.

### RPE-weighted effective tonnage

The repository uses

\[
L_{eff}=L_{set}\frac{RPE}{8}.
\]

This is an explicit project heuristic that makes RPE 8 neutral, RPE 9 count 12.5% more and missing RPE fall back to plain tonnage. It should not be described as a validated physiological load measure.

### Epley estimated 1RM

\[
\widehat{1RM}=w\left(1+\frac{r}{30}\right).
\]

The project tracks the best estimate per session. PR markers require a **strict** improvement over all prior estimates; ties with an existing best are not new PRs.

## 3. Recent-to-baseline load ratio (historically ACWR)

The short and long histories are exponentially weighted means:

\[
A_t=EWMA_{7}(L_t),\qquad C_t=EWMA_{28}(L_t),
\]

and the displayed ratio is

\[
R_t=\frac{A_t}{C_t}.
\]

The repository retains the function name `acwr` for API continuity, but uses the quantity only as a descriptive comparison of recent and longer-horizon load.

### Why the old interpretation was retired

The original README called 0.8–1.3 a “sweet spot” and treated values above 1.3 as injury-risk spikes. That is too strong. Methodological critiques identify ratio artefacts, confounding, discretisation and causal-interpretation problems, and specifically caution against using ACWR manipulation as an established injury-prevention prescription. A 2025 meta-analysis reports observational associations but also substantial heterogeneity and explicitly recommends caution.

Accordingly:

- no universal safe zone is claimed;
- the plot has a neutral reference at ratio = 1 rather than a green safety band;
- report language says recent load is below/similar/above its longer baseline;
- any branching in `recommend.py` is labelled an illustrative project heuristic.

References: PMID 32502973, PMID 32125672, PMID 41029871.

## 4. Causal personal baselines

For metric \(x_t\), the rolling z-score uses **prior observations only**:

\[
z_t=\frac{x_t-\mu_{t-1}^{(28)}}{s_{t-1}^{(28)}}.
\]

Today's value is excluded from the mean and SD. This avoids self-normalisation and guarantees that changing future observations cannot change a historical feature.

The readiness score

\[
R^{ready}_t=z^{HRV}_t-0.5z^{RHR}_t+0.5z^{sleep}_t
\]

is a transparent project heuristic. The weights are not clinically calibrated.

## 5. Lag diagnostic

For lag \(k\), the repo reports the Pearson correlation between load on day \(t\) and HRV z-score on day \(t+k\). This is an association diagnostic. It does not adjust for confounding and is not called a causal effect estimate.

## 6. Next-day prediction

Target:

\[
y_t=z^{HRV}_{t+1}.
\]

Features at day \(t\):

- causal load z-score;
- sleep z-score;
- HRV z-score;
- recent-to-baseline load ratio.

The model is ridge regression. Evaluation uses expanding time-ordered folds.

### One-day embargo

Without a gap, the last training observation has target \(z^{HRV}_{t+1}\), while the first test row uses that same \(z^{HRV}_{t+1}\) as its current-HRV feature. `TimeSeriesSplit(gap=1)` removes that target/feature overlap at fold boundaries.

### Fold-local scaling

`StandardScaler` is fitted inside each training fold through a scikit-learn pipeline. Test-period information therefore does not set feature means or variances.

### Baselines

The primary baseline is persistence:

\[
\hat y_{t+1}=z^{HRV}_{t}.
\]

The audit also compares the full model with a recovery-only model that omits load-derived features.

### 100-world known-truth result

Because the synthetic generator deliberately includes a load-to-HRV pathway, this experiment is a falsifiable code-validation test.

Across 100 worlds:

- median MAE improvement vs persistence = 28.25%;
- 5th–95th percentile = 22.68–34.21%;
- full model beats persistence in 100/100 worlds;
- full model beats recovery-only in 100/100 worlds;
- median lag-1 load/HRV correlation = -0.424;
- standardised load coefficient is negative in 100/100 worlds.

These values are not real-athlete performance claims.

## 7. Banister-style fitness-fatigue model

The model is

\[
P(t)=p_0+k_1F(t)-k_2D(t),
\]

where

\[
F(t)=\sum_s w(s)e^{-(t-s)/\tau_1},\qquad
D(t)=\sum_s w(s)e^{-(t-s)/\tau_2}.
\]

The implementation constrains \(\tau_1>\tau_2\), preserving the intended slower fitness / faster fatigue ordering.

### Optimisation

The fit uses nonlinear least squares with deterministic multi-start initialisation. Multi-start reduces dependence on one initial guess; it does not prove the parameters are identifiable.

### Known-truth identifiability audit

Truth across 100 structured-load worlds:

- \(\tau_1=35\) days;
- \(\tau_2=9\) days.

**Dense clean scenario:** observations every 2 days, zero observation noise. Both taus are recovered exactly. This checks the implementation under an ideal inverse problem.

**Weekly noisy scenario:** observations every 7 days, Gaussian noise SD = 3 performance units.

Results:

- \(\tau_1\) median 35.73 d; 5th–95th percentile 20.48–48.48 d;
- \(\tau_2\) median 8.31 d; 5th–95th percentile 4.34–19.87 d;
- median relative error 13.0% and 27.6%, respectively;
- median observed-point RMSE 2.63 units.

The point is not that those exact intervals generalise. The point is that adding realistic sparsity/noise turns an exactly recoverable code test into a practically weak parameter-estimation problem.

That conclusion is consistent with published warnings that fitness-fatigue parameters can be ill-conditioned, highly correlated and sensitive to modelling choices (PMID 16608765; PMID 35320776; PMID 39881202).

## 8. The end-to-end synthetic athlete is not a parameter-recovery benchmark

The synthetic athlete uses a planned latent impulse to generate strength dynamics, while the analysis later feeds realised RPE-weighted tonnage into the Banister fit. Those quantities are related but not identical.

Therefore the end-to-end fitted parameters should not be compared directly with the generator's latent taus and interpreted as recovery error. In the checked-in run the fitted gains are ~18.97 and 20.0, with the second at its optimisation bound, while the fitted taus are ~16.37 and 14.50 days. The correct interpretation is: the curve is an illustrative fit whose parameters require identifiability diagnostics.

This is precisely why `audit.py` contains a separate, controlled inverse-problem benchmark.

## 9. Taper calculation

`form_curve` treats load at index \(t\) as affecting the state at \(t+1\). For a completed load history of length \(n\), the first genuine rest day is therefore model index \(n\). `taper_days` searches those future rest days and returns 1 for the first day after the final impulse.

A model-derived peak day is conditional on the fitted model and future zero-load assumption. It is not a training prescription.

## 10. Data-quality assumptions

### Hevy

The loader fails on:

- missing required columns;
- exact duplicate set rows;
- negative weights or reps;
- RPE outside [1, 10] when supplied.

### Apple Health sleep

Sleep records can overlap across stages/sources. The parser unions overlapping asleep intervals before summing duration. Full timezone-aware timestamps are used for duration; local end-date text determines the morning bucket.

## 11. Planning heuristic

`recommend.py` exists to demonstrate transparent decision logic. It exposes every reason and emits:

`illustrative planning heuristic; not injury or medical advice`

No branch is presented as a clinically validated safety rule or optimal programme.

## 12. Public-claim hierarchy

Strongest public claims:

1. the leakage-aware pipeline recovers the deliberately planted recovery coupling across repeated synthetic worlds;
2. Banister parameter recovery is exact in an ideal known-truth setting but becomes much less precise under weekly noisy observation;
3. the project explicitly retired an overconfident ACWR injury/sweet-spot interpretation after methodological review.

Claims to avoid:

- “ACWR predicts my injury risk”;
- “0.8–1.3 is safe”;
- “the model tells me exactly how many days to taper”;
- “HRV can be predicted 28% better in real life”;
- “the fitted taus are my physiological fitness and fatigue constants”.

## Notes moved from the README

- Hevy ingestion: `ingest.py` validates the set export, maps exercises to broad muscle groups and computes plain tonnage, a heuristic RPE-weighted `effective_tonnage`, session-level Epley e1RM trajectories, strict estimated-PR improvements and weekly muscle-group volume. Missing RPE falls back to 8, so the weighted metric then equals plain tonnage. Negative loads or reps, impossible RPE values, duplicate set rows and missing required columns fail loudly.
- Apple Health: `export.xml` is streamed, not loaded whole. The parser extracts HRV SDNN, resting heart rate and asleep intervals. Overlapping asleep records are merged before summing, using timezone-aware timestamps, and overnight sleep is attributed to the local date on which it ends.
- Load ratio: the function keeps the name `acwr` for compatibility. I read it as a descriptive recent-to-baseline load signal. I do not claim that 0.8 to 1.3 is a sweet spot, that a ratio above 1.3 causes injury or that managing the ratio prevents injury.
- Readiness score: `HRV z - 0.5 x resting-HR z + 0.5 x sleep z`, each relative to the previous 28 days only. It is a project heuristic and not a validated physiological index.
- Next-day prediction: a ridge model on today's causal load z-score, sleep z-score, HRV z-score and load ratio, evaluated with expanding `TimeSeriesSplit`, a one-day embargo and fold-local scaling. The baseline is persistence.
- Banister fit: `tau_slow > tau_fast` is enforced and the optimiser uses deterministic multi-start nonlinear least squares. In the checked-in end-to-end run one gain sits at its bound, which is a diagnostic warning.
- Planning heuristic: the session suggestion combines the readiness heuristic, the load ratio and muscle-group volume against the preceding 28-day weekly average. Each branch returns its reasons and a status string saying it is an illustrative heuristic and not injury or medical advice. The weekly report uses neutral descriptive wording; the earlier "sweet spot / spike territory / cut volume" language was removed.
- The CLI fails if real-data mode is only partly specified, and does not fall back to synthetic data.
- Test coverage (19 tests): Epley and PR logic, Hevy validation and RPE weighting, Apple Health parsing and sleep merging, causal rolling baselines, ordered Banister recovery and noisy fits, the embargoed prediction, future-safe load features, non-overlapping volume baselines, recommender branches, report wording and small regression versions of both audits.

References:

- Impellizzeri FM et al. Acute:Chronic Workload Ratio: Conceptual Issues and Fundamental Pitfalls. International Journal of Sports Physiology and Performance (2020). PMID 32502973.
- Wang C et al. Acute to chronic workload ratio (ACWR) for predicting sports injury risk: a systematic review and meta-analysis (2025). PMID 41029871.
- Vanrenterghem J et al. The Fitness-Fatigue Model: What's in the Numbers? International Journal of Sports Physiology and Performance (2022). PMID 35320776.
- Hellard P et al. Assessing the limitations of the Banister model in monitoring training. Journal of Sports Sciences (2006). PMID 16608765.
- Statistical flaws of the fitness-fatigue sports performance prediction model. PMID 39881202.
