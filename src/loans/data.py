"""Load SBA 7(a) loans into one row per disbursed loan with only approval-time inputs.

Outcome: `default5` = 1 if SBA charged the loan off within 5 years (60 months) of approval.
A loan still active ("EXEMPT") or paid in full has not been charged off, so the outcome is known
for every disbursed loan approved at least 5 years before the data date (2026-06-30).

Never loaded: borrower name and street address. Never inputs (unknown at approval, or later status):
loan status and its dates, charge-off amount, first disbursement date, secondary-market sale, and
the current servicing bank (loans are reassigned after origination).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
AS_OF = pd.Timestamp("2026-06-30")
HORIZON_MONTHS = 60
FIRST_FY = 2010  # interest rate is missing for every loan before FY2009 and 11% of FY2009
LAST_FY = 2020   # FY2020 approvals end 2020-09-30; their 5-year window closes before AS_OF

READ_COLS = ["ApprovalFY", "ApprovalDate", "GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate",
             "FixedorVariableInterestInd", "TermInMonths", "NaicsCode", "FranchiseCode", "ProjectState",
             "BusinessType", "BusinessAge", "LoanStatus", "ChargeOffDate", "GrossChargeOffAmount",
             "RevolverStatus", "JobsSupported", "CollateralInd", "ProcessingMethod"]
OUTCOME_FIELDS = ["LoanStatus", "ChargeOffDate", "GrossChargeOffAmount"]
NEVER_READ = ["BorrName", "BorrStreet", "PaidInFullDate", "FirstDisbursementDate", "SoldSecMrktInd",
              "BankName", "BankFDICNumber", "BankNCUANumber", "BankStreet", "BankCity", "BankState",
              "BankZip", "LocationID"]

NUMERIC = ["log_amount", "guarantee_share", "interest_rate", "term_months", "jobs", "log_jobs"]
CATEGORICAL = ["sector", "state", "business_type", "business_age", "processing", "fixed_rate", "revolver",
               "collateral", "franchise", "real_estate_term"]
FEATURES = NUMERIC + CATEGORICAL


def business_age(s: pd.Series) -> pd.Series:
    """Harmonize to categories that mean the same before and after SBA's FY2018 change.

    Before FY2018 existing businesses were split into 2-5 and 5+ years; from FY2018 they are one
    "more than 2 years" group plus "change of ownership", so both eras collapse to "existing".
    """
    m = {
        "Startup, Loan Funds will Open Business": "startup",
        "New, Less than 1 Year old": "under_2y",
        "New Business or 2 years or less": "under_2y",
        "Less than 3 years old but at least 2": "existing",
        "Less than 4 years old but at least 3": "existing",
        "Less than 5 years old but at least 4": "existing",
        "Existing or more than 2 years old": "existing",
        "Existing, 5 or more years": "existing",
        "Change of Ownership": "existing",
    }
    return s.map(m).fillna("unanswered")


def processing(s: pd.Series) -> pd.Series:
    m = {"SBA Express Program": "express", "Preferred Lenders Program": "preferred_lender",
         "7a General": "general", "Certified Lenders Program": "general"}
    out = s.map(m)
    out[s.str.contains("Community|Small Loan Advantage|Rural|Patriot|Gulf", na=False)] = "pilot_initiative"
    out[s.str.contains("CAPLine|WCP|Export|International|EXIM", na=False)] = "working_capital_trade"
    return out.fillna("other")


def load(first_fy: int = FIRST_FY, last_fy: int = LAST_FY) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    cols = ", ".join(f'"{c}"' for c in READ_COLS)
    d = con.execute(f"""
        SELECT {cols} FROM read_csv('{RAW}/FOIA_7a_*.csv', header=true, all_varchar=true, union_by_name=true)
        WHERE TRY_CAST(ApprovalFY AS INTEGER) BETWEEN {first_fy} AND {last_fy}
    """).fetchdf()
    log = {"loans_in_window": len(d)}
    d["LoanStatus"] = d["LoanStatus"].str.replace(" ", "", regex=False)
    disbursed = ~d["LoanStatus"].isin(["CANCLD", "COMMIT"])
    log["cancelled_or_undisbursed"] = int((~disbursed).sum())
    d = d[disbursed].copy()

    d["fy"] = d["ApprovalFY"].astype(int)
    d["approval_date"] = pd.to_datetime(d["ApprovalDate"])
    co = pd.to_datetime(d["ChargeOffDate"], errors="coerce")
    months = (co.dt.year - d["approval_date"].dt.year) * 12 + (co.dt.month - d["approval_date"].dt.month)
    d["charged_off"] = (d["LoanStatus"] == "CHGOFF").astype(int)
    d["default5"] = ((d["charged_off"] == 1) & (months <= HORIZON_MONTHS)).astype(int)
    d["chargeoff_amount"] = pd.to_numeric(d["GrossChargeOffAmount"], errors="coerce").fillna(0.0)
    assert (d["approval_date"] + pd.DateOffset(months=HORIZON_MONTHS) <= AS_OF).all()

    amount = pd.to_numeric(d["GrossApproval"], errors="coerce")
    guar = pd.to_numeric(d["SBAGuaranteedApproval"], errors="coerce")
    d["amount"] = amount
    d["log_amount"] = np.log(amount)
    d["guarantee_share"] = guar / amount
    d["interest_rate"] = pd.to_numeric(d["InitialInterestRate"], errors="coerce")
    d["term_months"] = pd.to_numeric(d["TermInMonths"], errors="coerce")
    d["jobs"] = pd.to_numeric(d["JobsSupported"], errors="coerce")
    d["log_jobs"] = np.log1p(d["jobs"].clip(lower=0))
    d["sector"] = d["NaicsCode"].str[:2].fillna("unknown")
    d["state"] = d["ProjectState"].fillna("unknown")
    d["business_type"] = d["BusinessType"].fillna("unknown").str.lower()
    d["business_age"] = business_age(d["BusinessAge"])
    d["processing"] = processing(d["ProcessingMethod"])
    d["fixed_rate"] = (d["FixedorVariableInterestInd"] == "F").astype(int).astype(str)
    d["revolver"] = (d["RevolverStatus"].isin(["Y", "1"])).astype(int).astype(str)
    d["collateral"] = (d["CollateralInd"] == "Y").astype(int).astype(str)
    d["franchise"] = d["FranchiseCode"].notna().astype(int).astype(str)
    # Terms of 20 years or more are real-estate loans.
    d["real_estate_term"] = (d["term_months"] >= 240).astype(int).astype(str)
    bad = d[NUMERIC].isna().any(axis=1)
    log["missing_numeric_dropped"] = int(bad.sum())
    d = d[~bad]
    log["analysis_loans"] = len(d)
    keep = ["fy", "approval_date", "amount", "default5", "charged_off", "chargeoff_amount"] + FEATURES
    return d[keep].reset_index(drop=True), log
