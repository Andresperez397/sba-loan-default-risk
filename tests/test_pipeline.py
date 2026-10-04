from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loans import data, metrics, models  # noqa: E402

spec = importlib.util.spec_from_file_location("run_models", ROOT / "scripts" / "run_models.py")
run_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_models)
HAS_DATA = any(data.RAW.glob("FOIA_7a_*.csv"))


def fake_loans(n=8000, fys=(2010, 2011, 2015), seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "fy": rng.choice(fys, n),
            "amount": np.exp(rng.normal(11.8, 1, n)),
            "guarantee_share": rng.choice([0.5, 0.75, 0.85], n),
            "interest_rate": rng.uniform(4, 10, n),
            "jobs": rng.integers(0, 40, n),
            "sector": rng.choice(["44", "72", "54", "23"], n),
            "state": rng.choice(["TX", "CA", "NY", "FL"], n),
            "business_type": rng.choice(["corporation", "individual"], n),
            "business_age": rng.choice(["existing", "startup", "under_2y", "unanswered"], n),
            "processing": rng.choice(["express", "preferred_lender"], n),
            "fixed_rate": rng.choice(["0", "1"], n),
            "revolver": rng.choice(["0", "1"], n),
            "collateral": rng.choice(["0", "1"], n),
            "franchise": rng.choice(["0", "1"], n),
        }
    )
    d["log_amount"] = np.log(d["amount"])
    d["log_jobs"] = np.log1p(d["jobs"])
    lp = (
        -3.5
        + 0.25 * (d["interest_rate"] - 7)
        + 0.6 * (d["business_age"] == "startup")
        - 0.3 * (d["log_amount"] - 11.8)
    )
    d["default5"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    d["chargeoff_amount"] = d["default5"] * d["amount"] * 0.7
    return d


def test_no_leaky_or_never_read_field_is_an_input():
    assert not set(data.FEATURES) & set(data.OUTCOME_FIELDS + data.NEVER_READ + data.LEAKY_INPUTS)
    assert "term_months" not in data.FEATURES and "real_estate_term" not in data.FEATURES
    assert not {"BorrName", "BorrStreet"} & set(data.READ_COLS)


def test_business_age_harmonized_across_the_fy2018_change():
    s = pd.Series(
        [
            "Existing, 5 or more years",
            "Less than 3 years old but at least 2",
            "Existing or more than 2 years old",
            "Change of Ownership",
            "New, Less than 1 Year old",
            "New Business or 2 years or less",
            "Startup, Loan Funds will Open Business",
            "Unanswered",
            None,
        ]
    )
    assert data.business_age(s).tolist() == ["existing"] * 4 + ["under_2y"] * 2 + [
        "startup",
        "unanswered",
        "unanswered",
    ]


def test_processing_groups():
    s = pd.Series(
        [
            "SBA Express Program",
            "Preferred Lenders Program",
            "Certified Lenders Program",
            "Community Advantage Initiative",
            "Working Capital CAPLine",
            "Something new",
        ]
    )
    assert data.processing(s).tolist() == [
        "express",
        "preferred_lender",
        "general",
        "pilot_initiative",
        "working_capital_trade",
        "other",
    ]


def test_test_cohort_outcomes_cannot_change_its_predictions():
    d = fake_loans()
    train, test = d[d["fy"] <= 2010], d[d["fy"] == 2015]
    flipped = test.assign(default5=1 - test["default5"])
    old_grid, old_iters = models.HGB_GRID, models.HGB_ITERS
    models.HGB_GRID, models.HGB_ITERS = models.HGB_GRID[:1], [100, 200]
    try:
        for name, f in [("rate", models.fit_predict_rate), ("hgb", models.fit_predict_hgb)]:
            assert np.array_equal(f(train, test)[0], f(train, flipped)[0]), name
        assert np.array_equal(
            models.fit_predict_scorecard(train, test)[0], models.fit_predict_scorecard(train, flipped)[0]
        )
    finally:
        models.HGB_GRID, models.HGB_ITERS = old_grid, old_iters


def test_scorecard_points_reproduce_its_probabilities():
    d = fake_loans()
    sc = models.WoEScorecard(1.0).fit(d)
    pts = sc.points_table()
    total = np.zeros(len(d))
    for col in data.FEATURES:
        lookup = pts[pts["feature"] == col].set_index("bin")["points"]
        total += sc._bin(col, d[col]).map(lookup).to_numpy()
    factor = models.PDO / np.log(2)
    offset = models.BASE_SCORE - factor * np.log(models.BASE_ODDS)
    p_from_points = 1 / (1 + np.exp((total - offset) / factor))
    assert np.allclose(p_from_points, sc.predict(d), atol=1e-8)


def test_scorecard_bins_come_from_training_data_only():
    d = fake_loans()
    train, test = d[d["fy"] <= 2011], d[d["fy"] == 2015]
    a = models.WoEScorecard(1.0).fit(train).transform(test)
    shifted = test.assign(interest_rate=test["interest_rate"] + 50)  # far outside the training range
    b = models.WoEScorecard(1.0).fit(train).transform(shifted)
    col = data.FEATURES.index("interest_rate")
    assert np.allclose(b[:, col], b[0, col]) and np.isfinite(a).all()


def test_boosting_never_stops_early():
    m = models.hgb(models.HGB_GRID[0], 100)
    d = fake_loans()
    cats = {c: sorted(d[c].unique()) for c in data.CATEGORICAL}
    m.fit(models.hgb_frame(d, cats), d["default5"])
    assert m.n_iter_ == 100 and not m.early_stopping


def test_metrics():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 4000), rng.uniform(size=4000)
    assert metrics.auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert metrics.ks(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    same = pd.Series(rng.normal(size=5000))
    assert run_models.psi(same, same.sample(frac=1, random_state=1), True) < 0.01
    assert run_models.psi(same, same + 2, True) > 0.25


def test_policy_curve_approves_the_lowest_scores():
    P = pd.DataFrame(
        {
            "fy": [2015] * 10,
            "amount": [100.0] * 10,
            "guarantee_share": [0.5] * 10,
            "default5": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "chargeoff_amount": [0.0] * 8 + [80.0, 80.0],
        }
    )
    score = np.arange(10, dtype=float)  # the two defaults have the highest risk scores
    c = run_models.policy_curve(P, score, [0.8, 1.0]).set_index("approve_share")
    assert c.loc[0.8, "default_rate"] == 0 and c.loc[0.8, "lender_loss_per_100"] == 0
    assert c.loc[1.0, "default_rate"] == pytest.approx(0.2)
    assert c.loc[1.0, "lender_loss_per_100"] == pytest.approx(100 * 2 * 80 * 0.5 / 1000)


@pytest.mark.skipif(not HAS_DATA, reason="raw data not downloaded")
def test_outcome_window_and_rules_on_real_data():
    d, log = data.load(2015, 2015)
    assert set(d["default5"].unique()) <= {0, 1}
    assert (d["default5"] <= d["charged_off"]).all()  # early charge-off implies charge-off
    assert d["interest_rate"].min() >= 2
    assert log["analysis_loans"] == len(d)
