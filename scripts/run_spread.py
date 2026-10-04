"""Exploratory (DEVIATIONS.md): interest rate as a spread over the prime rate on the approval date.

Same validation design as run_models.py; refits only the rate-only baseline and the scorecard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loans import data, metrics, models  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_models import DERIVED, N_BOOT, OUT, TEST_FY, boot_indices, policy_curve, psi  # noqa: E402


def with_spread(d: pd.DataFrame) -> pd.DataFrame:
    prime = pd.read_csv(data.RAW / "DPRIME.csv", parse_dates=["observation_date"], na_values=".").dropna()
    prime = prime.rename(columns={"observation_date": "date", "DPRIME": "prime"}).sort_values("date")
    m = pd.merge_asof(
        d[["approval_date"]].reset_index().sort_values("approval_date"),
        prime,
        left_on="approval_date",
        right_on="date",
        direction="backward",
    ).set_index("index")
    out = d.copy()
    out["prime"] = m["prime"].reindex(d.index).to_numpy()
    out["spread"] = out["interest_rate"] - out["prime"]
    return out


def main() -> None:
    d = with_spread(data.load()[0])
    assert d["prime"].notna().all()
    ds = d.assign(interest_rate=d["spread"])  # the models read the column named interest_rate
    P = pd.concat([pd.read_parquet(DERIVED / f"cohort_{s}.parquet") for s in TEST_FY], ignore_index=True)
    sp_rate, sp_card = [], []
    for s in TEST_FY:
        train, test = ds[ds["fy"] <= s - 5], ds[ds["fy"] == s]
        assert np.allclose(test["amount"].to_numpy(), P.loc[P["fy"] == s, "amount"].to_numpy())
        sp_rate.append(models.fit_predict_rate(train, test)[0])
        sp_card.append(models.fit_predict_scorecard(train, test)[0])
    P["spread only"] = np.concatenate(sp_rate)
    P["scorecard with spread"] = np.concatenate(sp_card)
    y = P["default5"].to_numpy()
    cols = ["M1 interest rate", "spread only", "M2 scorecard", "scorecard with spread"]
    boot = {c: [] for c in cols}
    for idx in boot_indices(P["fy"].to_numpy(), N_BOOT):
        for c in cols:
            boot[c].append(metrics.auc(y[idx], P[c].to_numpy()[idx]))
    res = {
        "auc": {c: metrics.auc(y, P[c]) for c in cols},
        "auc_by_cohort": {
            c: {int(s): metrics.auc(y[P["fy"] == s], P.loc[P["fy"] == s, c]) for s in TEST_FY} for c in cols
        },
        "scorecard_spread_minus_scorecard": {
            "est": metrics.auc(y, P["scorecard with spread"]) - metrics.auc(y, P["M2 scorecard"]),
            "ci": list(metrics.ci(np.array(boot["scorecard with spread"]) - np.array(boot["M2 scorecard"]))),
        },
        "scorecard_minus_spread_only": {
            "est": metrics.auc(y, P["M2 scorecard"]) - metrics.auc(y, P["spread only"]),
            "ci": list(metrics.ci(np.array(boot["M2 scorecard"]) - np.array(boot["spread only"]))),
        },
        "scorecard_with_spread_minus_spread_only": {
            "est": metrics.auc(y, P["scorecard with spread"]) - metrics.auc(y, P["spread only"]),
            "ci": list(metrics.ci(np.array(boot["scorecard with spread"]) - np.array(boot["spread only"]))),
        },
        "spread_only_minus_rate_only": {
            "est": metrics.auc(y, P["spread only"]) - metrics.auc(y, P["M1 interest rate"]),
            "ci": list(metrics.ci(np.array(boot["spread only"]) - np.array(boot["M1 interest rate"]))),
        },
    }
    train_all = d[d["fy"] <= 2015]
    res["psi"] = {
        int(s): {
            "rate": psi(train_all["interest_rate"], d.loc[d["fy"] == s, "interest_rate"], True),
            "spread": psi(train_all["spread"], d.loc[d["fy"] == s, "spread"], True),
        }
        for s in TEST_FY
    }
    pc = policy_curve(P, P["scorecard with spread"].to_numpy(), [0.7, 0.75, 0.8, 0.9])
    res["policy_scorecard_with_spread"] = pc.to_dict(orient="records")
    res["mean_spread_by_fy"] = d.groupby("fy")["spread"].mean().round(3).to_dict()
    with open(OUT / "exploratory_spread.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
