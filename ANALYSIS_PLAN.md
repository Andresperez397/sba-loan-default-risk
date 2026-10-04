# Analysis plan (frozen before any outcome model is fit)

Written 2026-10-04 after the data audit (`DATA_AUDIT.md`) and before any model of loan default was fit. Any later change is logged in `DEVIATIONS.md` with a date and reason.

**Unit:** one disbursed SBA 7(a) loan approved FY2010–FY2020 (515,427 loans).
- **Cleaning rule:** loans with an initial interest rate below 2% or a term under 1 month are dropped as entry errors (about 680 loans).

**Outcome:** `default5`, charged off within 60 months of approval.
**Inputs:** only the approval-time fields listed in `DATA_AUDIT.md`.

## Q1. How well can early charge-off be predicted for loans approved after the model was built, and does a model beat the lender's own price for risk?

**Validation honors what a lender could know.**
- **Test cohorts:** FY2015–FY2020, six in all.
- **Training:** test cohort *s* is scored by a model trained only on cohorts FY2010 through *s*−5. Their five-year outcomes were known by the start of *s*.
- **Tuning:** hyperparameters are tuned by 5-fold cross-validation within the training cohorts, the same way for every learner.

**Models**

| Model | What it is |
|---|---|
| M0 | Training default rate (constant) |
| M1 | Interest rate only: logistic regression on a natural spline (df 4) of the initial rate. The rate is the lender's price for risk, so this is the bar a model must clear. |
| M2 | **Scorecard:** logistic regression on weight-of-evidence (WoE) bins of every input |
| M3 | **Gradient boosting:** scikit-learn `HistGradientBoostingClassifier` on the same inputs |

**Scorecard (M2):**
- **Bins:** numeric inputs are cut into up to 10 quantile bins, and categorical levels with fewer than 500 training loans are pooled into "other". Bins are fixed on training data.
- **WoE values:** estimated with a smoothing of 0.5 events and non-events per bin.
- **Fit:** L2-penalized logistic regression on the WoE columns, with C tuned over 8 log-spaced values from 1e-3 to 1e1.
- **Points:** the fit is converted to points with the industry convention of 600 points at 50:1 odds and 20 points to double the odds.

**Gradient boosting (M3):**
- **Grid:** 8 settings, max leaf nodes {15, 63} × learning rate {0.05, 0.1} × min samples per leaf {100, 1000}.
- **Iterations:** 100–800, chosen by cross-validation in steps of 100.
- **Settings:** `early_stopping=False`, categorical inputs handled natively, fixed seed.

**Metrics,** per test cohort and pooled over FY2015–FY2020 (loan-weighted):
- **Primary:** AUC (and Gini = 2·AUC − 1).
- **Secondary:**
  - the Kolmogorov–Smirnov (KS) statistic
  - log loss and Brier score
  - calibration: logistic recalibration intercept and slope, and observed against predicted default rate by decile.

**Uncertainty:** bootstrap of loans within each test cohort (1,000 resamples), paired across models.

**Decision rules**
- **A model beats the lender's risk price** if its pooled AUC exceeds M1's with a 95% interval above zero.
- **Boosting earns its complexity** if its AUC exceeds the scorecard's with a 95% interval above zero. Otherwise the scorecard is preferred for the policy and app, because it is transparent and adverse-action reasons can be read off it.

## Q2. What is a better approval policy worth?

**The lender's view.** The lender's loss on a charged-off loan is its unguaranteed share: charge-off amount × (1 − guarantee share). Both are in the data.

**Policies compared on each test cohort.** Each approves the safest *q*% of applicants (*q* from 50% to 100% in steps of 5) by:
- the preferred model's predicted default probability
- interest rate (lower rate is treated as safer)
- random order (the baseline).

**Outcomes per policy and *q*:**
- early charge-off rate among approved loans
- lender loss per $100 approved, from realized charge-offs (early ones only, to match the outcome)
- dollars approved.

**Headline:** at a 90% approval rate (decline the riskiest 10%), the reduction in early charge-offs and in lender loss per $100 against declining by interest rate. Intervals use the same bootstrap.

## Q3. Model risk: does the model hold up across segments and through COVID?

For the preferred model on the test cohorts:
1. **Population stability index (PSI)** between training and each test cohort, for the score and for each input. The conventional flags are 0.10 (watch) and 0.25 (act).
2. **Calibration by cohort.** Observed against predicted early charge-off rate per test cohort. FY2020 is expected to be over-predicted because of CARES Act payment relief, which is a named, pre-stated expectation.
3. **Discrimination and calibration by segment:** AUC and observed/predicted ratio by
   - sector (top 10 by volume)
   - loan size band (under $50k, $50–150k, $150–500k, $500k+)
   - business age, processing method, business type
   - region (Census regions from project state).

   Segments with fewer than 50 defaults in the test cohorts are reported but flagged.

## Deliverables

- **README:** figures and pooled results.
- **Two-page PDF summary.**
- **R Shiny app** with three parts:
  - **Loan desk:** enter a loan and get the scorecard points, the default probability and the top reasons.
  - **Policy:** the approval-rate trade-off.
  - **Monitoring:** PSI and segment calibration.
- **Static fallback.**

The app uses the scorecard exported as a points table, so no Python is needed to run it.

## Software

- **Python 3.11:** pandas, numpy, scikit-learn, statsmodels, DuckDB for loading.
- **R and Shiny** for the app.
- **Versions** are pinned in `requirements.txt` and `app/manifest.json`.
- **Seeds:** all random steps are seeded.
