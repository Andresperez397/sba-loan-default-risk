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
