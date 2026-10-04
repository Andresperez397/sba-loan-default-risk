"""Default models for Q1: constant, interest-rate-only, a WoE scorecard and gradient boosting.

The scorecard and boosting are tuned the same way: 5-fold cross-validation within the training cohorts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import data
from .metrics import logloss

C_GRID = np.logspace(-3, 1, 8)
HGB_GRID = [dict(max_leaf_nodes=m, learning_rate=lr, min_samples_leaf=msl)
            for m in (15, 63) for lr in (0.05, 0.1) for msl in (100, 1000)]
HGB_ITERS = list(range(100, 801, 100))
N_BINS = 10
MIN_LEVEL = 500
SMOOTH = 0.5
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50  # 600 points at 50:1 good:bad odds, 20 points to double the odds


def folds(y, k=5, seed=0):
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


class NaturalSpline:
    """Natural cubic spline basis (ESL eq. 5.4-5.5) with knots at training quantiles."""

    def __init__(self, df: int):
        self.df = df

    def fit(self, x):
        self.knots = np.unique(np.quantile(x, np.linspace(0, 1, self.df + 1)))
        return self

    def transform(self, x):
        k, last = self.knots, self.knots[-1]

        def d(j):
            return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - last, 0, None) ** 3) / (last - k[j])

        return np.column_stack([x] + [d(j) - d(len(k) - 2) for j in range(len(k) - 2)])


def fit_predict_const(train, test):
    return np.full(len(test), train["default5"].mean()), {}


def fit_predict_rate(train, test):
    sp = NaturalSpline(4).fit(train["interest_rate"].to_numpy())
    m = LogisticRegression(C=1e6, max_iter=2000).fit(sp.transform(train["interest_rate"].to_numpy()),
                                                      train["default5"])
    return m.predict_proba(sp.transform(test["interest_rate"].to_numpy()))[:, 1], {}


class WoEScorecard:
    """Weight-of-evidence bins on every input, then penalized logistic regression, then points."""

    def __init__(self, c: float = 1.0):
        self.c = c

    def _bin(self, col: str, x: pd.Series) -> pd.Series:
        if col in data.NUMERIC:
            idx = np.searchsorted(self.edges[col], x.to_numpy(), side="right")
            return pd.Series(idx, index=x.index).astype(str)
        return x.where(x.isin(self.levels[col]), "other").astype(str)

    def fit_bins(self, d: pd.DataFrame) -> WoEScorecard:
        self.edges, self.levels, self.woe = {}, {}, {}
        y = d["default5"].to_numpy()
        for col in data.FEATURES:
            if col in data.NUMERIC:
                q = np.quantile(d[col], np.linspace(0, 1, N_BINS + 1)[1:-1])
                self.edges[col] = np.unique(q)
            else:
                vc = d[col].value_counts()
                self.levels[col] = set(vc[vc >= MIN_LEVEL].index)
            b = self._bin(col, d[col])
            g = pd.DataFrame({"b": b.to_numpy(), "y": y}).groupby("b")["y"].agg(["sum", "size"])
            bad, good = g["sum"] + SMOOTH, g["size"] - g["sum"] + SMOOTH
            self.woe[col] = np.log((good / good.sum()) / (bad / bad.sum()))  # positive = safer than average
        return self

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in data.FEATURES:
            b = self._bin(col, d[col])
            cols.append(b.map(self.woe[col]).fillna(0.0).to_numpy())
        return np.column_stack(cols)

    def fit(self, d: pd.DataFrame) -> WoEScorecard:
        self.fit_bins(d)
        self.lr = LogisticRegression(C=self.c, max_iter=2000).fit(self.transform(d), d["default5"])
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.transform(d))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """Points per bin: score = sum of points; higher = safer (lower default probability)."""
        factor = PDO / np.log(2)
        offset = BASE_SCORE - factor * np.log(BASE_ODDS)
        beta, alpha, n = self.lr.coef_[0], self.lr.intercept_[0], len(data.FEATURES)
        rows = []
        for j, col in enumerate(data.FEATURES):
            for b, w in self.woe[col].items():
                # log-odds of default = alpha + sum beta_j * WoE_j; good:bad log-odds is its negative.
                pts = factor * (-(alpha / n) - beta[j] * w) + offset / n
                rows.append({"feature": col, "bin": b, "woe": float(w), "points": float(pts)})
        t = pd.DataFrame(rows)
        edges = {c: list(map(float, e)) for c, e in self.edges.items()}
        t.attrs["edges"] = edges
        return t


def fit_predict_scorecard(train, test):
    y = train["default5"].to_numpy()
    split = folds(y)
    # WoE bins are refit inside each fold, so validation outcomes never shape the inputs used to tune C.
    fold_x = []
    for tr, va in split:
        sc = WoEScorecard().fit_bins(train.iloc[tr])
        fold_x.append((sc.transform(train.iloc[tr]), sc.transform(train.iloc[va])))
    scores = []
    for c in C_GRID:
        ll = []
        for (tr, va), (xtr, xva) in zip(split, fold_x, strict=True):
            m = LogisticRegression(C=c, max_iter=2000).fit(xtr, y[tr])
            ll.append(logloss(y[va], m.predict_proba(xva)[:, 1]))
        scores.append(np.mean(ll))
    c = C_GRID[int(np.argmin(scores))]
    sc = WoEScorecard(c).fit(train)
    return sc.predict(test), {"C": float(c), "cv_logloss": float(min(scores))}, sc


def hgb_frame(d: pd.DataFrame, cats: dict) -> pd.DataFrame:
    x = d[data.FEATURES].copy()
    for c in data.CATEGORICAL:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return x


def hgb(params, max_iter, warm_start=False):
    # early_stopping=False: scikit-learn otherwise stops silently on an internal split for n > 10,000.
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, warm_start=warm_start,
                                          categorical_features="from_dtype", random_state=0, **params)


def fit_predict_hgb(train, test):
    cats = {c: sorted(train[c].unique()) for c in data.CATEGORICAL}
    x, y = hgb_frame(train, cats), train["default5"].to_numpy()
    rows = []
    for i, params in enumerate(HGB_GRID):
        ll = {it: [] for it in HGB_ITERS}
        for tr, va in folds(y):
            m = hgb(params, HGB_ITERS[0], warm_start=True)
            for it in HGB_ITERS:
                m.set_params(max_iter=it).fit(x.iloc[tr], y[tr])
                ll[it].append(logloss(y[va], m.predict_proba(x.iloc[va])[:, 1]))
        rows += [{"setting": i, "iters": it, "cv_logloss": float(np.mean(v))} for it, v in ll.items()]
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["cv_logloss"].idxmin()]
    m = hgb(HGB_GRID[int(best["setting"])], int(best["iters"])).fit(x, y)
    tuning = {**HGB_GRID[int(best["setting"])], "iters": int(best["iters"]),
              "cv_logloss": float(best["cv_logloss"])}
    return m.predict_proba(hgb_frame(test, cats))[:, 1], tuning
