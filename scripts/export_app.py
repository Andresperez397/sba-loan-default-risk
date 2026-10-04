"""Export the scorecard and summary tables for the Shiny app (app/data/).

The app's scorecard is the pre-specified M2 scorecard refit on every cohort with a known five-year outcome
(FY2010-FY2020), with its penalty tuned by the same cross-validation. Performance claims come from the
held-out cohorts in run_models.py, not from this refit.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loans import data, models  # noqa: E402

APP = ROOT / "app" / "data"
TABLES = ROOT / "reports" / "tables"


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)
    d, _ = data.load()
    _, tuning, sc = models.fit_predict_scorecard(d, d.head(1))
    pts = sc.points_table()
    factor = models.PDO / np.log(2)
    card = {
        "fit_on": "FY2010-FY2020 approvals (all cohorts with a known five-year outcome)",
        "C": tuning["C"],
        "factor": factor,
        "offset": models.BASE_SCORE - factor * np.log(models.BASE_ODDS),
        "numeric": data.NUMERIC,
        "categorical": data.CATEGORICAL,
        "edges": {c: list(map(float, e)) for c, e in sc.edges.items()},
        "levels": {c: sorted(map(str, v)) for c, v in sc.levels.items()},
        "points": pts.to_dict(orient="records"),
        # Points for an input bin the training data never saw (weight of evidence 0).
        "neutral_points": float(
            factor * (-(sc.lr.intercept_[0] / len(data.FEATURES)))
            + (models.BASE_SCORE - factor * np.log(models.BASE_ODDS)) / len(data.FEATURES)
        ),
        "choices": {c: sorted(d[c].astype(str).unique()) for c in data.CATEGORICAL},
        "numeric_ranges": {
            c: [float(d[c].quantile(0.01)), float(d[c].median()), float(d[c].quantile(0.99))]
            for c in ["amount", "guarantee_share", "interest_rate", "jobs"]
        },
    }
    with open(APP / "scorecard.json", "w") as f:
        json.dump(card, f, indent=1)
    for name in [
        "q1_pooled.csv",
        "q1_by_cohort.csv",
        "q1_deciles.csv",
        "q2_policy_curves.csv",
        "q3_segments.csv",
        "summary.json",
        "exploratory_spread.json",
    ]:
        shutil.copy(TABLES / name, APP / name)
    print("exported scorecard with", len(pts), "bins; C =", tuning["C"])


if __name__ == "__main__":
    main()
