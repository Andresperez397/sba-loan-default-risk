# Data audit

Written 2026-10-04, before any model of loan default was fit. Every number here comes from `scripts/data_audit.py` (saved to `reports/tables/data_audit.json`).

## Source and license

- **Data:** U.S. Small Business Administration, 7(a) loan-level FOIA data, snapshot "as of June 30, 2026". It covers every 7(a) loan approved since FY1991.
- **License:** "U.S. Government Works" (public domain).
- **Pinning:** SBA republishes the files each quarter under new names. `scripts/fetch_data.py` pins this snapshot by SHA-256 in `data/manifest.csv`.
- **Privacy:** the raw files name borrowers, and many borrowers are individuals or sole proprietors. Borrower name and street address are never read. Raw data is not committed, and only aggregates are published.

## Coverage and the outcome

- **Loans:** 1,624,422 7(a) loans approved from FY2000 to the as-of date.
- **Status:**

| Status | Loans | Meaning |
|---|---|---|
| Paid in full | 923,144 | |
| Active ("EXEMPT") | 295,634 | Status withheld under FOIA |
| Cancelled | 199,124 | Never disbursed |
| Charged off | 185,397 | |
| Committed, undisbursed | 21,123 | Never disbursed |

**Outcome.** The outcome is `default5`: SBA charged the loan off within 60 months of approval.
- **Why it is always known:** an active ("EXEMPT") loan has, by definition, not been charged off. So the outcome is known for every disbursed loan approved at least five years before the data date.
- **Which loans count:** cancelled and undisbursed loans are excluded, because there was no loan.

**Charge-offs lag defaults.** SBA charges a loan off after collection and liquidation, so charge-off dates trail the borrower's default.
- **Timing:** for FY2010–FY2019 approvals, the median charge-off comes 53–59 months after approval.
- **Coverage:** only 52–60% of a cohort's eventual charge-offs fall within 60 months.

So `default5` measures **early charge-off**, not every loss, and the README says so. It is still the right target for a lender's decision, and it is the only one that is fully observed for recent cohorts.

## Window: FY2010 to FY2020

- **Why the window starts at FY2010:** the initial interest rate is missing for every loan approved FY2000–FY2008 and for 11% of FY2009. The rate is the lender's own price for risk, and it is needed both as an input and as a benchmark.
- **Why it ends at FY2020:** FY2020 approvals end on 2020-09-30, so their 60-month window closes on 2025-09-30, before the data date. FY2021 windows would still be open.

**Analysis set.**
- **In the window:** 588,049 loans approved FY2010–FY2020.
- **Removed:** 72,622 cancelled or undisbursed.
- **Remaining:** 515,427 disbursed loans, then 514,984 after dropping 443 with an initial rate under 2% (the cleaning rule; see `DEVIATIONS.md`). The table below is after that rule.

| FY | Loans | Charged off within 5 years | Ever charged off (to date) | Mean rate |
|---|---|---|---|---|
| 2010 | 39,903 | 5.0% | 9.2% | 6.45% |
| 2012 | 38,883 | 3.3% | 6.3% | 6.23% |
| 2014 | 45,889 | 3.6% | 6.4% | 6.04% |
| 2016 | 56,736 | 3.9% | 7.2% | 6.27% |
| 2018 | 54,168 | 4.7% | 7.9% | 7.29% |
| 2019 | 45,601 | 4.0% | 6.7% | 7.92% |
| 2020 | 36,368 | 2.6% | 4.0% | 6.47% |

**Two era effects to keep in mind:**
- **The 2008–09 recession:** charge-offs reached 27–33% for FY2006–FY2008 cohorts. Those cohorts are outside the window, but they show how much default rates move with the economy.
- **COVID relief:** the FY2020 cohort's low rate (2.6%) coincides with the CARES Act, under which SBA made six months of payments on 7(a) loans in 2020–21. A model trained on earlier cohorts should be expected to over-predict defaults for FY2020.

## Inputs (known when the loan is approved)

| Group | Fields |
|---|---|
| Loan terms | amount (log), SBA guarantee share, initial interest rate, fixed or variable, revolving line, collateral (term was dropped; see the last section) |
| Business | 2-digit NAICS sector, project state, business type (corporation, individual, partnership), business age, franchise, jobs supported (self-reported, log) |
| Program | processing method, grouped into express, preferred lender, general, pilot initiatives, working capital/trade |

**Not used as inputs:**
- **Unknown at approval, or the outcome itself:**
  - loan status, paid-in-full and charge-off dates, and charge-off amount
  - first disbursement date
  - the secondary-market sale flag, which is set after origination
- **Lender identity and location:** these fields describe the bank the loan is *currently* assigned to, and loans are transferred after origination, including during collections.
- **Borrower name and street address:** never read.

## Recording changes and data quality

1. **Business-age categories changed in FY2018.** Before, existing businesses were split into 2–5 and 5+ years. From FY2018 there is one "more than 2 years" group plus "change of ownership". Both eras are collapsed to startup / under 2 years / existing / unanswered.
   - **FY2018–FY2019 coding drift:** "unanswered" rises to 15–19% (6–10% in other years), and "under 2 years" is absent in FY2019 while the new codes phased in. "Unanswered" is kept as its own level.
2. **Implausible terms.** 443 loans have an initial interest rate below 2% (14 at exactly 0%), and 236 have a 0-month term. These look like entry errors, and they are dropped under a rule fixed in the plan.
3. **Processing methods come and go.** Pilot programs start and end (for example, Small Loan Advantage FY2011–FY2017 and Patriot Express FY2007–FY2014). They are grouped so that categories exist across the window.
4. **Interest rates moved with the market.** Average rates rise from about 6.0% (FY2014) to 7.9% (FY2019) as the prime rate rose. The rate is partly a price for risk and partly the market level.
5. **Loss given default:** among FY2010–FY2020 loans that were charged off, the median charge-off amount is 78% of the approved amount.

## Leakage risks carried into the analysis plan

1. **Outcome-related fields** (status, dates, charge-off amount, sale flag, current bank) are never inputs. A test enforces this.
2. **Look-ahead in outcomes.** A lender deciding in fiscal year *s* knows five-year outcomes only for loans approved by *s*−5. The validation design respects that: a test cohort is scored only by a model trained on cohorts at least five years older.
3. **Random splits** would mix cohorts that went through different economies. Validation is by approval cohort.

## Loan term is overwritten for loans that went bad (found 2026-10-04, after the plan froze)

The first pipeline check gave implausibly high accuracy, and the cause was the term field. It is the contractual term for loans that performed, but SBA overwrites it for loans that went bad.

| Group | Share with a standard term (60, 84, 120, 240, 300 months) |
|---|---|
| Charged off within 5 years | 2.6% |
| Not charged off within 5 years | 73.9% |

Charged-off loans instead show terms like 67, 68 or 73 months, and 72% of loans with a term under one month were charged off. On its own, term separates defaults with an AUC of 0.86, against at most 0.65 for any other input.

Because term is not known at approval in the form the data records it, term and the real-estate flag built from it are not inputs. The other inputs were screened the same way, and none shows the pattern: approved amounts and guarantee shares cluster at the same round values for defaults and non-defaults. See `DEVIATIONS.md`.
