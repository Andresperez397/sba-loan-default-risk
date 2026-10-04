"""Scoring rules, calibration and the game-cluster bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

CLIP = 0.001


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def logloss_i(y, p):
    p = clip(np.asarray(p, float))
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logloss(y, p) -> float:
    return float(logloss_i(y, p).mean())


def brier_i(y, p):
    return (np.asarray(y, float) - np.asarray(p, float)) ** 2


def auc(y, p) -> float:
    y = np.asarray(y)
    r = rankdata(p)
    n1 = float((y == 1).sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration(y, p) -> dict:
    """Logistic recalibration (slope 1, intercept 0 = calibrated) and 10-bin expected calibration error."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)  # positional indexing below; a pandas Series would index by label
    lp = np.log(clip(p) / (1 - clip(p)))
    if np.ptp(lp) == 0:  # a constant forecast has no slope
        slope = float("nan")
    else:
        slope = np.asarray(sm.GLM(y, np.column_stack([np.ones_like(lp), lp]),
                                  family=sm.families.Binomial()).fit().params)[1]
    recal = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    intercept = np.asarray(recal.params)[0]
    bins = pd.qcut(rankdata(p, method="ordinal"), 10, labels=False)
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    ece = float((g["n"] * (g["y"] - g["p"]).abs()).sum() / len(df))
    return {"cal_intercept": float(intercept), "cal_slope": float(slope), "ece": ece}


def score(y, p) -> dict:
    return {"logloss": logloss(y, p), "brier": float(brier_i(y, p).mean()), "auc": auc(y, p),
            **calibration(y, p)}


def boot_weights(cluster, stratum, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Clusters x n_boot counts of how often each cluster is drawn, resampling within each stratum."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"c": cluster, "s": stratum}).drop_duplicates("c")
    w = np.zeros((len(g), n_boot), dtype=np.int32)
    for s in g["s"].unique():
        idx = np.flatnonzero(g["s"].to_numpy() == s)
        draws = rng.integers(0, len(idx), size=(len(idx), n_boot))
        for b in range(n_boot):
            w[idx, b] = np.bincount(draws[:, b], minlength=len(idx))
    return pd.DataFrame(w, index=g["c"].to_numpy())


def boot_mean(values, cluster, w: pd.DataFrame) -> np.ndarray:
    """Pooled per-row mean of `values` under each bootstrap resample."""
    s = pd.Series(values).groupby(np.asarray(cluster)).agg(["sum", "size"]).reindex(w.index)
    return (w.to_numpy().T @ s["sum"].to_numpy()) / (w.to_numpy().T @ s["size"].to_numpy())


def ci(x) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def ks(y, p) -> float:
    """Kolmogorov-Smirnov statistic: largest gap between the score distributions of defaults and the rest."""
    y = np.asarray(y)
    order = np.argsort(p)
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))
