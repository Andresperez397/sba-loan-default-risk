# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen (commit fb27a17) is logged here, dated, with its reason.

## Analysis

- **2026-10-04, loan term removed as an input (leakage found in the first model run):**
  - **How it was found:** a pipeline check on one split (train FY2010, test FY2015) gave AUCs of 0.89 (scorecard) and 0.95 (boosting), far above what small-business credit models reach. The scorecard put most of its weight on term.
  - **What is wrong with term:** SBA overwrites the term of loans that went bad.
    - **Not charged off:** 74% of loans that were not charged off within five years carry a standard contractual term (60, 84, 120, 240 or 300 months).
    - **Charged off:** only 2.6% of those charged off do. Their terms take values like 67, 68 or 73 months, and a term under one month belongs to a charged-off loan 72% of the time.
    - **On its own:** term reaches an AUC of 0.86. No other input exceeds 0.65.
  - **Response:** `term_months` and the 20-year real-estate flag built from it are removed from the inputs. The term is still loaded, so the audit can show the problem.
  - **What was seen:** only the results of that single check; no reported result uses term.
  - **Cost:** the real-estate indicator goes with it. The README says so.
- **2026-10-04, cleaning rule changed:** the planned rule dropped loans with a rate under 2% *or* a term under 1 month. The term part is removed, because those terms are overwrites on charged-off loans (72% charged off), and dropping them would remove defaults. Only the rate rule remains: 443 loans, with a 2.7% charge-off rate, consistent with entry errors.
- **2026-10-04, added exploratory analysis (Q3), after seeing results:** interest rate as a spread over the prime rate.
  - **Why:** Q3's stability check flagged the absolute interest rate as the most unstable input. Its PSI between the training cohorts and the test cohorts was 0.82 in FY2018 and 1.62 in FY2019, because the prime rate rose from 3.25% to 5.50%. An absolute rate mixes the lender's price for risk with the market rate level.
  - **What it does:** the check replaces the rate with its spread over the Wall Street Journal prime rate on the approval date (FRED series DPRIME, public, no account). It refits the scorecard and the rate-only baseline with the same validation design.
  - **Labelled:** exploratory. The pre-specified results above are reported unchanged.
