# Changelog

## 0.4.0, defensibility audit

- Reframed ACWR as a descriptive recent-to-baseline load ratio; removed the unsupported universal `0.8–1.3 sweet spot` / injury-risk interpretation.
- Made recovery z-scores causal by comparing each morning with the previous 28 days only.
- Closed a one-step time-series CV boundary leak with a one-day gap and fold-local feature scaling.
- Added 100-world known-truth prediction and Banister identifiability audits.
- Constrained Banister time constants to preserve slow-fitness / fast-fatigue ordering and added deterministic multi-start fitting.
- Fixed the taper-day indexing convention and strict PR detection.
- Made the recommendation output explicitly an illustrative planning heuristic, not medical or injury-prevention advice.
- Hardened Hevy validation and Apple Health sleep handling, including timezone-aware duration and overlap merging.
- Made CLI data mode fail closed instead of silently switching to synthetic data when one real input is missing.
- Added `docs/METHODOLOGY.md` and expanded regression/statistical tests.

## 0.3.0, update round 9
A weekly report generator: last seven days of load by muscle against the four week average, the ACWR flag in words, e1RM estimates on the key lifts, and one concrete recommendation, written to markdown.

## 0.2.0, update round 2
Added RPE-weighted effective load (a set at RPE 9 counts 9/8 of its kilos, missing RPE falls back to plain tonnage), now the default load metric for ACWR, prediction and the Banister fit.

## 0.1.0
Initial release.
