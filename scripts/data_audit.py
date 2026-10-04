"""Data audit (run before any outcome model). Writes reports/tables/data_audit.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loans import data, metrics  # noqa: E402

con = duckdb.connect()
con.execute(f"""CREATE TABLE l AS SELECT ApprovalFY, ApprovalDate, LoanStatus, ChargeOffDate,
                InitialInterestRate, BusinessAge, GrossChargeOffAmount, GrossApproval
                FROM read_csv('{data.RAW}/FOIA_7a_*.csv', header=true, all_varchar=true,
                              union_by_name=true)""")


def q(sql):
    return con.execute(sql).fetchdf()


out: dict = {}
out["all_loans_fy2000_on"] = int(q("select count(*) n from l")["n"][0])
out["status_counts"] = q("select replace(LoanStatus,' ','') s, count(*) n from l group by 1 order by 2 desc"
                         ).set_index("s")["n"].to_dict()
out["interest_rate_missing_share_by_fy"] = q(
    "select ApprovalFY fy, round(avg((InitialInterestRate is null)::int), 4) v from l group by 1 order by 1"
).set_index("fy")["v"].to_dict()
co = q("""select ApprovalFY fy, count(*) chargeoffs,
          quantile_cont(date_diff('month', ApprovalDate::date, ChargeOffDate::date), 0.5) median_months,
          avg((date_diff('month', ApprovalDate::date, ChargeOffDate::date) <= 60)::int) share_within_60
          from l where replace(LoanStatus,' ','')='CHGOFF' group by 1 order by 1""")
out["chargeoff_timing_by_fy"] = co.set_index("fy").round(4).to_dict(orient="index")
out["business_age_categories_first_fy"] = q(
    "select BusinessAge c, min(ApprovalFY) first_fy, max(ApprovalFY) last_fy, count(*) n "
    "from l group by 1 order by 4 desc"
).to_dict(orient="records")
out["chargeoff_amount_over_approval_median"] = float(q(
    """select median(GrossChargeOffAmount::double / GrossApproval::double) v from l
       where replace(LoanStatus,' ','')='CHGOFF' and ApprovalFY::int between 2010 and 2020""")["v"][0])

d, log = data.load()
out["load_log"] = log
out["by_fy"] = d.groupby("fy").agg(
    loans=("default5", "size"), default5=("default5", "mean"), ever_charged_off=("charged_off", "mean"),
    interest_rate=("interest_rate", "mean"), median_amount=("amount", "median"),
).round(4).to_dict(orient="index")
out["categorical_levels"] = {c: d[c].value_counts().to_dict() for c in data.CATEGORICAL if c != "state"}
out["numeric_ranges"] = d[data.NUMERIC].describe().T[["min", "50%", "max"]].round(4).to_dict(orient="index")
# Term is overwritten for loans that went bad (found after the plan froze; DEVIATIONS.md).
std = d["term_months"].isin([60, 84, 120, 240, 300])
out["term_standard_share"] = {"default5": float(std[d["default5"] == 1].mean()),
                              "no_default5": float(std[d["default5"] == 0].mean())}
zero = q("""select avg((replace(LoanStatus,' ','')='CHGOFF')::int) v, count(*) n from read_csv(
            '""" + str(data.RAW) + """/FOIA_7a_*.csv', header=true, all_varchar=true, union_by_name=true)
            where ApprovalFY::int between 2010 and 2020
            and replace(LoanStatus,' ','') not in ('CANCLD','COMMIT') and TermInMonths::double < 1""")
out["term_under_1_month"] = {"loans": int(zero["n"][0]), "charged_off_share": float(zero["v"][0])}
single = {}
early = d[d["fy"] <= 2012]
for col in data.FEATURES + ["term_months"]:
    b = pd.qcut(early[col].rank(method="first"), 20, labels=False) if col in data.NUMERIC + ["term_months"] \
        else early[col]
    single[col] = round(metrics.auc(early["default5"], early.groupby(b)["default5"].transform("mean")), 3)
out["single_input_auc_fy2010_2012"] = single
out["never_read"] = data.NEVER_READ
out["outcome_fields_never_inputs"] = data.OUTCOME_FIELDS
assert not set(data.FEATURES) & set(data.OUTCOME_FIELDS + data.NEVER_READ)

(ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)
with open(ROOT / "reports" / "tables" / "data_audit.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
for k in ("all_loans_fy2000_on", "status_counts", "load_log", "chargeoff_amount_over_approval_median",
          "term_standard_share", "term_under_1_month", "single_input_auc_fy2010_2012"):
    print(k, out[k])
print(pd.DataFrame(out["by_fy"]).T)
print(pd.DataFrame(out["chargeoff_timing_by_fy"]).T.tail(12))
print(d["business_age"].groupby(d["fy"]).value_counts(normalize=True).unstack().round(3))
