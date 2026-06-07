# nifty50-iv-surface-reconstruction

A two-step pipeline for reconstructing missing implied volatility values across a NIFTY50 options surface, combining volatility smile interpolation with a gradient boosting residual correction ensemble.

**Best public score: MSE = 0.0000389346**

---

## The problem

Options traders do not quote prices in rupees — they quote implied volatility (IV). A complete IV surface maps how this number varies across strike prices and time, and it is the foundation of every options pricing and risk management system in practice.

The dataset is a 975 × 28 matrix of five-minute IV observations across 13 trading days (January 7–27, 2026) for 28 NIFTY50 option strikes: 14 calls from 25200 to 26500, and 14 puts from 23800 to 25100, all expiring January 27, 2026. About 19% of the matrix is missing, concentrated in illiquid deep-OTM strikes and early-session bars. The task is to fill every gap as accurately as possible, measured by MSE against a hidden test set (30% public leaderboard, 70% private).

---

## Approach

The solution is a two-step pipeline grounded in how options markets actually work.

### Volatility Smile Interpolation

At any fixed point in time, all 14 call IVs lie on a smooth U-shaped curve called the volatility smile. Near-the-money options have the lowest IV; it rises symmetrically on both wings. If 10 of the 14 strikes are observed at a given timestamp, I can fit this curve and read off the 4 missing values with reasonable accuracy.

The interpolator blends quadratic and linear fits. Quadratic splines can overshoot badly at the wings where data is sparsest, so the blend is asymmetric:

- **Interior strikes** (target between observed strikes): 60% quadratic + 40% linear
- **Wing strikes** (extrapolation beyond observed range): 85% linear + 15% quadratic
- **Normal day final prediction**: 90% spline + 10% exponentially-weighted moving average of recent history for stability
- **Expiry day**: 100% spline — past IV levels are irrelevant when gamma is exploding

This stage uses only data from the same row. No past or future timestamps are accessed anywhere, making it completely lookahead-free.

### Gradient Boosting Residual Correction

The spline has systematic errors that correlate with moneyness zone, time of day, and recent IV dynamics. A LightGBM + CatBoost ensemble (50/50) learns to predict these residuals from 47 engineered features. The final prediction formula is:

```
IV_final = spline_pred + α(zone) × (0.50 × LightGBM + 0.50 × CatBoost)
```

Three separate models handle the three distinct regimes in the data.

---

## Why three separate models

**CE calls vs PE puts.** Call residuals have 3.6× higher standard deviation than put residuals — visible directly in the training data. A single shared model over-indexes on call patterns and systematically underfits puts. Separating them immediately improved the leaderboard score.

**Expiry day.** On January 27 (DTE = 0), gamma explodes as time value evaporates. IV can legitimately jump from 0.5 to 5+ in the final 30 minutes as each 5-minute bar represents an ever-larger fraction of remaining time. A model trained on the previous 12 normal trading days has never seen this dynamic and makes large errors on expiry. The dedicated expiry model uses `num_leaves=7` and `reg_lambda=4.0` — much shallower and more regularised than the CE/PE models — because its training set has only ~1,350 rows and is highly non-stationary.

---

## Zone-based alpha for CE calls

A controlled error analysis — hiding 20% of observed cells and measuring spline error by moneyness zone — revealed a strong pattern in where the spline fails:

| Zone | \|log-moneyness\| range | Spline MSE × 10⁶ |
|---|---|---|
| Near ATM | < 0.02 | 0.55 |
| Moderate OTM | 0.02 – 0.05 | 2.81 |
| Deep OTM | ≥ 0.05 | 0.14 |

The moderate OTM zone sits at the inflection point of the smile where curvature changes most rapidly. The spline systematically under-corrects there. Raising the correction weight to 0.90 specifically for that zone gave a confirmed leaderboard improvement. The final alpha values used in production:

| Region | Weight (α) |
|---|---|
| CE near ATM (\|lm\| < 0.02) | 0.70 |
| CE moderate OTM (0.02 – 0.05) | 0.90 |
| CE deep OTM (\|lm\| ≥ 0.05) | 0.70 |
| PE puts (uniform across all zones) | 0.60 |
| Expiry day | 0.45 |

---

## Feature engineering

47 features across five groups, all strictly lookahead-free.

**Smile position (5 features)**
Log-moneyness, moneyness, volatility-standardised moneyness, strike rank, is_call. These place the cell in the right position on the smile curve and normalise across different IV regimes.

**Term structure (5 features)**
DTE (days to expiry), bar index (0–74 within the session), fractional session position, trading day number, early-day flag. DTE is the single most important feature — its relationship with gamma is highly nonlinear and becomes extreme on expiry day.

**Surface shape (16 features)**
ATM IV level (calls, puts, and average), mean and minimum IV by type, observed count by type, smile skew (mean PE minus mean CE), surface dispersion, call and put curvature coefficients, wing ratio (max IV divided by ATM IV), and boundary gap (max PE IV minus min CE IV). All computed from the same timestamp row — zero temporal access.

**Time-series lags (8 features)**
`iv_lag1`, `iv_lag2`, rolling means over 3 and 10 bars, velocity (first difference between consecutive bars), and filled versions of each that substitute the spline estimate when the lag is unavailable at the start of each trading day.

**Market dynamics (2 features)**
5-minute log return of NIFTY50 (computed with `shift(1)` so it always references the previous bar), and its interaction with `is_call`. The leverage effect means downside market moves lift call IV disproportionately — this asymmetry is a real, exploitable signal.

**Interaction terms (5 features)**
`lag1_spline_ratio` (how much does the recent IV level deviate from the current spline estimate?), `lm × spline_confidence`, `lm × neighbour_count`, `surface_dispersion × strike_rank`, `log(DTE) × |lm|`.

The lag buffer uses a strict read-before-append pattern: the history buffer is read first — recording lag values for the current bar — and only then is the current observation appended to the buffer. This is the mechanism that guarantees all lag features are lookahead-free. Current information cannot appear in its own lag.

---

## Score progression

Every row is a confirmed Kaggle submission, not a local estimate.

| Configuration | Public MSE |
|---|---|
| Spline baseline only | 0.0001631 |
| + LightGBM residual correction | 0.0000425 |
| + Separate CE / PE models | 0.0000403 |
| + 5-minute return and leverage interaction feature | 0.0000402 |
| + CatBoost added to ensemble | 0.0000398 |
| + Zone-based alpha for CE calls | 0.0000396 |
| + XGBoost (LGB 50% / CB 25% / XGB 25%) | 0.0000395 |
| **LGB + CatBoost 50/50, moderate OTM alpha = 0.90, expiry alpha = 0.45** | **0.0000380** |

---

## What did not work

Most of the development time went into ideas that failed. They are documented here because the reasons they failed are instructive.

**PCHIP interpolation** showed +1.2% improvement in local testing but −12.4% on the Kaggle leaderboard. Real missing cells cluster in illiquid strikes where PCHIP has no nearby anchor points. The local simulation hides random observed cells and creates artificially dense neighbourhoods that do not reflect this structure.

**Bivariate splines across time and strike** — catastrophic locally (−1180% MSE). IV is not smooth across time at a five-minute scale. The bivariate approach assumes joint smoothness in both dimensions simultaneously, which does not hold.

**GP / SVD / NMF matrix completion** — all worse than the simple quadratic/linear spline. These methods assume global low-rank structure across the full matrix. A single-expiry surface over 13 days does not have this structure.

**Temporal scaling for expiry day** — showed +11% improvement in local simulation but doubled the Kaggle error. The root cause was a bug: the scaling prior used the last observed value for each strike, which on the first expiry bar comes from a previous normal trading day. Scaling a 0.15 non-expiry IV by a 4× gamma factor produces 0.60, when the true expiry IV might be 0.18. The local simulation did not expose this because it always had same-day neighbours available.

**Three-model ensemble (LGB + CatBoost + XGBoost)** — marginally better than two models in some configurations, worse in others. The 50/50 LGB + CatBoost blend was consistently more stable.

**Multi-seed LGB averaging** — zero measured effect on either local validation or the leaderboard. LightGBM with fixed hyperparameters and a fixed `random_state` produces very similar trees across different seeds. There is not enough variance between seeds to benefit from averaging.

The common thread: the local simulation creates a fundamentally different missingness pattern from the real competition data. Real missing cells are in places where the entire neighbourhood is also sparse. Any method that requires nearby observed data to function correctly will look better locally than it actually is. This pattern repeated itself throughout development and is the primary reason local validation results should not be trusted for this particular problem.

---

## How to run

1. Upload `iv-surface-final.ipynb` to a new Kaggle notebook
2. Add the competition dataset: **Data → Add Data → finclub-open-project-26**
3. Set accelerator to **None** (CPU only — GPU is not needed)
4. Click **Session → Restart and Run All**
5. Runtime is approximately 20–25 minutes
6. Verify the final cell prints: `Rows: 5460 | All positive: True | No NaN: True`
7. Submit `/kaggle/working/submission.csv`

The notebook is fully deterministic. `random.seed(42)` and `np.random.seed(42)` are set at the top. All models use `random_state=42`. CE and PE models use fixed `n_estimators` with no early stopping. The expiry model uses a fixed interleaved validation split (`index % 5 == 0`) which is deterministic given the fixed seed. Running the notebook twice produces identical output.

---

## Dependencies

All packages are available on Kaggle's default Python 3.10 kernel. No additional installation is required.

```
pandas    >= 1.5
numpy     >= 1.23
scipy     >= 1.9
lightgbm  >= 3.3
catboost  >= 1.1
matplotlib >= 3.5
```

---

## Repository

```
nifty50-iv-surface-reconstruction/
├── iv-surface-final.ipynb    Main notebook — reproduces submission.csv exactly
└── README.md                 This file
```
