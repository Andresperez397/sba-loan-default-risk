"""Q1-Q3 (ANALYSIS_PLAN.md): default models on later cohorts, approval policies, and model risk checks.

Test cohort FY s is scored only by models trained on cohorts FY2010..s-5, whose five-year outcomes were
known when FY s began. Each cohort's predictions are checkpointed in data/derived/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loans import data, metrics, models  # noqa: E402

TEST_FY = list(range(2015, 2021))
MODELS = ["M0 constant", "M1 interest rate", "M2 scorecard", "M3 boosting"]
DERIVED, OUT = ROOT / "data" / "derived", ROOT / "reports" / "tables"
N_BOOT = 1000
REGION = {  # Census regions by project state
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, ss in REGION.items() for s in ss.split()}


def run_cohort(d: pd.DataFrame, s: int) -> None:
    ck = DERIVED / f"cohort_{s}.parquet"
    if ck.exists():
        return
    t0 = time.time()
    train, test = d[d["fy"] <= s - 5], d[d["fy"] == s]
    out = test[
        [
            "fy",
            "amount",
            "guarantee_share",
            "interest_rate",
            "default5",
            "chargeoff_amount",
            "sector",
            "state",
            "business_type",
            "business_age",
            "processing",
            "log_amount",
        ]
    ].copy()
    tuning = {}
    out["M0 constant"], _ = models.fit_predict_const(train, test)
    out["M1 interest rate"], _ = models.fit_predict_rate(train, test)
    out["M2 scorecard"], tuning["M2 scorecard"], _ = models.fit_predict_scorecard(train, test)
    out["M3 boosting"], tuning["M3 boosting"] = models.fit_predict_hgb(train, test)
    for m in MODELS:
        print(s, m, round(metrics.auc(test["default5"], out[m]), 4), flush=True)
    out.to_parquet(ck, index=False)
    with open(DERIVED / f"tuning_{s}.json", "w") as f:
        json.dump({"tuning": tuning, "n_train": len(train), "minutes": (time.time() - t0) / 60}, f, indent=2)


def boot_indices(fy: np.ndarray, n_boot: int, seed: int = 0):
    """Resample loans with replacement within each test cohort."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fy == s) for s in np.unique(fy)]
    for _ in range(n_boot):
        yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


def policy_curve(P: pd.DataFrame, score: np.ndarray, qs) -> pd.DataFrame:
    """Approve the lowest-risk q share of each cohort by `score`; report outcomes pooled over cohorts."""
    rows = []
    loss = P["default5"] * P["chargeoff_amount"] * (1 - P["guarantee_share"])
    for q in qs:
        keep = np.zeros(len(P), bool)
        for s in P["fy"].unique():
            idx = np.flatnonzero(P["fy"].to_numpy() == s)
            k = int(round(q * len(idx)))
            keep[idx[np.argsort(score[idx], kind="stable")[:k]]] = True
        a = P[keep]
        rows.append(
            {
                "approve_share": q,
                "default_rate": float(a["default5"].mean()),
                "lender_loss_per_100": float(100 * loss[keep].sum() / a["amount"].sum()),
                "dollars_approved": float(a["amount"].sum()),
            }
        )
    return pd.DataFrame(rows)


def psi(expected: pd.Series, actual: pd.Series, numeric: bool) -> float:
    if numeric:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, 11)[1:-1]))
        e = pd.Series(np.searchsorted(edges, expected, side="right")).value_counts(normalize=True)
        a = pd.Series(np.searchsorted(edges, actual, side="right")).value_counts(normalize=True)
    else:
        e, a = expected.value_counts(normalize=True), actual.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx, fill_value=0) + 1e-4, a.reindex(idx, fill_value=0) + 1e-4
    return float(((a - e) * np.log(a / e)).sum())


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d, log = data.load()
    for s in TEST_FY:
        run_cohort(d, s)
    P = pd.concat([pd.read_parquet(DERIVED / f"cohort_{s}.parquet") for s in TEST_FY], ignore_index=True)
    y = P["default5"].to_numpy()
    res: dict = {
        "load_log": log,
        "n_test": len(P),
        "tuning": {s: json.load(open(DERIVED / f"tuning_{s}.json")) for s in TEST_FY},
    }

    # Q1: discrimination and calibration, per cohort and pooled, with a paired loan bootstrap.
    by = [
        {
            "fy": s,
            "model": m,
            "n": int((P["fy"] == s).sum()),
            "default_rate": float(y[P["fy"] == s].mean()),
            "auc": metrics.auc(y[P["fy"] == s], P.loc[P["fy"] == s, m]),
            "ks": metrics.ks(y[P["fy"] == s], P.loc[P["fy"] == s, m].to_numpy()),
            "mean_predicted": float(P.loc[P["fy"] == s, m].mean()),
            **metrics.score(y[P["fy"] == s], P.loc[P["fy"] == s, m]),
        }
        for s in TEST_FY
        for m in MODELS
    ]
    pd.DataFrame(by).to_csv(OUT / "q1_by_cohort.csv", index=False)
    pred = {m: P[m].to_numpy() for m in MODELS}
    boot = {m: [] for m in MODELS}
    for idx in boot_indices(P["fy"].to_numpy(), N_BOOT):
        for m in MODELS[1:]:
            boot[m].append(metrics.auc(y[idx], pred[m][idx]))
    pooled = []
    for m in MODELS:
        row = {
            "model": m,
            "auc": metrics.auc(y, pred[m]),
            "ks": metrics.ks(y, pred[m]),
            **metrics.score(y, pred[m]),
        }
        if boot[m]:
            row["auc_lo"], row["auc_hi"] = metrics.ci(boot[m])
        row["gini"] = 2 * row["auc"] - 1
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "q1_pooled.csv", index=False)
    comps = {}
    for a, b in (
        ("M2 scorecard", "M1 interest rate"),
        ("M3 boosting", "M1 interest rate"),
        ("M3 boosting", "M2 scorecard"),
    ):
        dl = np.array(boot[a]) - np.array(boot[b])
        comps[f"{a} minus {b}"] = {
            "auc_diff": metrics.auc(y, pred[a]) - metrics.auc(y, pred[b]),
            "ci": list(metrics.ci(dl)),
            "interval_above_zero": bool(metrics.ci(dl)[0] > 0),
        }
    res["q1_comparisons"] = comps
    preferred = (
        "M3 boosting" if comps["M3 boosting minus M2 scorecard"]["interval_above_zero"] else "M2 scorecard"
    )
    res["preferred_model"] = preferred
    deciles = P.assign(
        dec=P.groupby("fy")[preferred].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
    )
    deciles.groupby("dec").agg(
        predicted=(preferred, "mean"), observed=("default5", "mean"), loans=("default5", "size")
    ).to_csv(OUT / "q1_deciles.csv")

    # Q2: approval policies.
    qs = np.round(np.arange(0.5, 1.0001, 0.05), 2)
    curves = pd.concat(
        [
            policy_curve(P, P[preferred].to_numpy(), qs).assign(policy="model"),
            policy_curve(P, P["interest_rate"].to_numpy(), qs).assign(policy="interest rate"),
        ]
    )
    curves.to_csv(OUT / "q2_policy_curves.csv", index=False)
    base = curves[(curves["policy"] == "model") & (curves["approve_share"] == 1.0)].iloc[0]

    def headline(PP):
        m = policy_curve(PP, PP[preferred].to_numpy(), [0.9]).iloc[0]
        r = policy_curve(PP, PP["interest_rate"].to_numpy(), [0.9]).iloc[0]
        return (r["default_rate"] - m["default_rate"], r["lender_loss_per_100"] - m["lender_loss_per_100"])

    point = headline(P)
    hb = np.array(
        [headline(P.iloc[idx].reset_index(drop=True)) for idx in boot_indices(P["fy"].to_numpy(), N_BOOT)]
    )
    q90 = curves[curves["approve_share"] == 0.9].set_index("policy")
    res["q2_headline_at_90pct"] = {
        "approve_all": {
            "default_rate": float(base["default_rate"]),
            "lender_loss_per_100": float(base["lender_loss_per_100"]),
        },
        "model": q90.loc["model"].to_dict(),
        "interest_rate": q90.loc["interest rate"].to_dict(),
        "default_rate_reduction_vs_rate": {"est": point[0], "ci": list(metrics.ci(hb[:, 0]))},
        "loss_per_100_reduction_vs_rate": {"est": point[1], "ci": list(metrics.ci(hb[:, 1]))},
    }

    # Q3: model risk - stability and segment performance of the preferred model.
    q3 = {"psi_score": {}, "psi_inputs": {}}
    train_all = d[d["fy"] <= 2015]
    for s in TEST_FY:
        te = d[d["fy"] == s]
        q3["psi_inputs"][s] = {c: psi(train_all[c], te[c], c in data.NUMERIC) for c in data.FEATURES}
        q3["psi_score"][s] = psi(
            P.loc[P["fy"] == TEST_FY[0], preferred], P.loc[P["fy"] == s, preferred], True
        )
    P["region"] = P["state"].map(STATE_REGION).fillna("Territories/other")
    P["size_band"] = pd.cut(
        P["amount"], [0, 50e3, 150e3, 500e3, np.inf], labels=["<50k", "50-150k", "150-500k", "500k+"]
    )
    top_sectors = P["sector"].value_counts().head(10).index
    seg_rows = []
    for col in ("region", "size_band", "business_age", "processing", "business_type", "sector"):
        for lev, g in P.groupby(col, observed=True):
            if col == "sector" and lev not in top_sectors:
                continue
            ok = g["default5"].nunique() == 2
            seg_rows.append(
                {
                    "segment": col,
                    "level": str(lev),
                    "loans": len(g),
                    "defaults": int(g["default5"].sum()),
                    "auc": metrics.auc(g["default5"], g[preferred]) if ok else np.nan,
                    "observed": float(g["default5"].mean()),
                    "predicted": float(g[preferred].mean()),
                    "flag_few_defaults": bool(g["default5"].sum() < 50),
                }
            )
    seg = pd.DataFrame(seg_rows)
    seg["obs_over_pred"] = seg["observed"] / seg["predicted"]
    seg.to_csv(OUT / "q3_segments.csv", index=False)
    res["q3"] = q3

    with open(OUT / "summary.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(
        pooled[["model", "auc", "auc_lo", "auc_hi", "ks", "logloss", "cal_slope", "cal_intercept"]].round(4)
    )
    print(
        json.dumps(
            {k: res[k] for k in ("q1_comparisons", "preferred_model", "q2_headline_at_90pct")},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
