
### Nifty50 IV Surface Reconstruction

**Finance Club IIT Roorkee — Open Project 2026**

Reconstructing missing implied volatility values across a NIFTY50 options surface using a two-stage spline + gradient boosting pipeline.

**Best public score: MSE = 0.0000380**

---

## The problem

Options prices are quoted in implied volatility (IV) rather than rupees. A full IV surface maps how this number varies across strike prices and time — it is the foundation of every options pricing and risk management system.

The dataset is a 975 × 28 matrix: 975 five-minute bars across 13 trading days (January 7–27, 2026) and 28 NIFTY50 option strikes (14 calls from 25200 to 26500, 14 puts from 23800 to 25100), all expiring January 27, 2026. About 19% of entries are missing — concentrated in illiquid deep-OTM strikes and early-session bars — and the task is to fill every gap. The evaluation metric is mean squared error on a hidden test set (30% public, 70% private).

---

## Solution overview

The pipeline has two stages.

**Stage 1 — Volatility smile interpolation.** At any given moment, all 14 call IVs lie on a smooth U-shaped curve called the volatility smile. If 10 of the 14 strikes are observed, I can fit that curve and read off the 4 missing values. This is a purely cross-sectional operation — only same-row data is used, so there is zero temporal lookahead.

The interpolator blends quadratic and linear fits. Quadratic splines can overshoot badly at the wings where data is sparse, so the blend is asymmetric: 60% quadratic + 40% linear for interior strikes, and 85% linear + 15% quadratic for extrapolation. The spline estimate is further blended with an exponentially-weighted moving average of each strike's recent history (90% spline, 10% EWMA on normal days; 100% spline on expiry day where past IV levels are irrelevant).

**Stage 2 — Gradient boosting residual correction.** The spline has systematic errors that correlate with moneyness zone, time of day, and recent IV dynamics. A LightGBM + CatBoost ensemble (50/50) predicts these residuals from 47 engineered features. The final prediction is:

IV_final = spline_pred + α(zone) × (0.50 × LightGBM + 0.50 × CatBoost)

Three separate models handle the three distinct regimes in the data: CE calls, PE puts, and expiry day.

---

## Why three separate models

**CE vs PE.** Call residuals have 3.6× higher standard deviation than put residuals. A single shared model would over-index on call patterns and systematically underfit puts.

**Expiry day.** On January 27 (DTE = 0), gamma explodes and IV can jump from 1.5 to 5+ in the final 30 minutes as each 5-minute bar represents a larger fraction of remaining time value. A model trained on the previous 12 normal trading days has never seen this dynamic. The expiry model uses `num_leaves=7` and `reg_lambda=4.0` — much shallower and more regularised — because the training set has only ~1,350 rows and is highly non-stationary.

---

## Zone-based correction alpha for CE calls

A controlled error analysis (hiding 20% of observed cells and measuring spline error by moneyness zone) revealed a strong pattern:

| Zone | |log-moneyness| | Spline MSE × 10⁶ |
|---|---|---|
| Near ATM | < 0.02 | 0.55 |
| Moderate OTM | 0.02 – 0.05 | 2.81 |
| Deep OTM | ≥ 0.05 | 0.14 |

The moderate OTM zone is the inflection point of the smile where curvature is highest — the spline systematically under-corrects there. Raising the correction alpha to 0.90 for that zone gave a confirmed leaderboard improvement.

Final alpha values:

| Region | Alpha |
|---|---|
| CE near ATM (|lm| < 0.02) | 0.70 |
| CE moderate OTM (0.02–0.05) | 0.90 |
| CE deep OTM (|lm| ≥ 0.05) | 0.70 |
| PE puts (all zones) | 0.60 |
| Expiry day | 0.45 |

---

## Features

47 features across five groups, all strictly lookahead-free.

* **Smile position (5):** log-moneyness, moneyness, volatility-standardised moneyness, strike rank, is_call. These tell the model where on the smile curve this cell sits.
* **Term structure (5):** DTE, bar index (0–74), fractional session position, trading day number, early-day flag. DTE is the most important single feature — the relationship between DTE and gamma is highly nonlinear.
* **Surface shape (16):** ATM IV level (calls, puts, and average), mean and minimum IV by type, observed count by type, smile skew (mean PE minus mean CE), surface dispersion, call and put curvature coefficients, wing ratio, boundary gap. All computed from the same row — zero temporal access.
* **Time-series lags (8):** iv_lag1, iv_lag2, rolling means over 3 and 10 bars, velocity (first difference), and filled versions of each that substitute the spline estimate when the lag is unavailable at the start of each day.
* **Market dynamics (2):** 5-minute log return of NIFTY50 (`shift(1)` — previous bar only), and its interaction with `is_call`. The leverage effect means downside market moves lift call IV more than put IV — this asymmetry is a genuine signal.
* **Interaction terms (5):** `lag1_spline_ratio` (how much does the recent level deviate from the current spline?), `lm × spline_confidence`, `lm × neighbour_count`, `surface_dispersion × rank`, `log(DTE) × |lm|`.

The lag buffer uses a read-before-append pattern: the history buffer is read (recording lag values) and only then is the current observation appended. This is the mechanism that guarantees lookahead freedom — current information cannot appear in its own lag features.

---

## Score progression

Each row represents a confirmed Kaggle submission, not a local estimate.

| Model | Public MSE |
|---|---|
| Spline baseline only | 0.0001631 |
| + LightGBM residual correction | 0.0000425 |
| + Separate CE / PE models | 0.0000403 |
| + spot_ret1 and leverage interaction | 0.0000402 |
| + CatBoost in ensemble | 0.0000398 |
| + Zone-based alpha for CE | 0.0000396 |
| + XGBoost (50/25/25 blend) | 0.0000395 |
| LGB + CatBoost 50/50, mod OTM alpha = 0.90, expiry alpha = 0.45 | **0.0000380** |

---

## What failed

It is worth documenting what did not work, because most of the development time went into these.

* **PCHIP interpolation** — showed +1.2% improvement locally but −12.4% on Kaggle. Real missing cells cluster in illiquid strikes where PCHIP has no nearby anchors; the local simulation did not reproduce this.
* **Bivariate splines across time and strike** — catastrophic locally (−1180% MSE). IV is not smooth across time at a fine scale.
* **GP / SVD / NMF matrix completion** — all worse than the simple spline. These methods assume global low-rank structure that does not hold for a single-expiry surface over 13 days.
* **Temporal scaling for expiry day** — +11% local improvement but doubled the error on Kaggle. The bug: the scaling prior used the last observed value for that strike, which on the first expiry bar is from a previous non-expiry day. Scaling a 0.15 normal-day IV by a 4× gamma factor produces garbage.
* **Three-model ensemble (LGB + CatBoost + XGBoost)** — marginally better than two-model in one configuration but worse in another. The 50/50 LGB + CatBoost blend was more consistent.
* **Multi-seed LGB averaging** — zero measured effect. LightGBM with fixed hyperparameters and fixed `random_state` produces very similar trees across seeds; there is not enough variance to benefit from averaging.

The common thread in every failure: the local simulation hides random observed cells and creates artificially dense neighbourhoods around every missing cell. Real missing cells are in places where the entire neighbourhood is also sparse. Any method that requires nearby data to work will look better locally than it actually is.

---

## How to run

1. Upload `iv_surface_REFACTORED.ipynb` to Kaggle
2. Add data source: **finclub-open-project-26**
3. Accelerator: None (CPU only, no GPU needed)
4. Session → Restart and Run All
5. Runtime: approximately 20–25 minutes
6. Verify last cell prints: `Rows: 5460 | All positive: True | No NaN: True`
7. Submit `/kaggle/working/submission.csv`

The notebook is fully deterministic: `random.seed(42)`, `np.random.seed(42)`, `random_state=42` on all models, fixed `n_estimators` for CE/PE (no early stopping), and a fixed interleaved split for the expiry model.

---

## Dependencies

All available on Kaggle's default Python 3.10 kernel — no extra installation needed.

pandas >= 1.5
numpy >= 1.23
scipy >= 1.9
lightgbm >= 3.3
catboost >= 1.1
matplotlib >= 3.5
