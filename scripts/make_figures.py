"""Report figures from reports/tables/ (run after run_models.py and run_spread.py)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
GREEN, GREY, AMBER, INK, GRID = "#1f6f5c", "#8a929b", "#c27c0e", "#16212b", "#e4e3df"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def auc_by_cohort():
    c = pd.read_csv(TAB / "q1_by_cohort.csv")
    sp = json.load(open(TAB / "exploratory_spread.json"))["auc_by_cohort"]["spread only"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m, col, ls, lab in (
        ("M1 interest rate", GREY, "-", "Interest rate alone"),
        ("M3 boosting", AMBER, "-", "Gradient boosting"),
        ("M2 scorecard", GREEN, "-", "Scorecard"),
    ):
        s = c[c["model"] == m]
        ax.plot(s["fy"], s["auc"], marker="o", color=col, ls=ls, label=lab, lw=2)
    fys = sorted(int(k) for k in sp)
    ax.plot(
        fys, [sp[str(f)] for f in fys], marker="o", color=GREY, ls="--", label="Rate spread over prime alone*"
    )
    ax.set_xlabel("Approval fiscal year (each scored by a model trained on loans 5+ years older)")
    ax.set_ylabel("AUC")
    ax.set_title("Predicting early charge-off on later loans")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.02, "*exploratory check", transform=ax.transAxes, ha="right", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_auc_by_cohort.png", dpi=200)
    plt.close(fig)


def policy():
    p = pd.read_csv(TAB / "q2_policy_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, lab in (
        (axes[0], "default_rate", "Early charge-off rate among approved loans"),
        (axes[1], "lender_loss_per_100", r"Lender loss per \$100 approved (\$)"),
    ):
        for pol, c, name in (
            ("interest rate", GREY, "Decline highest interest rates"),
            ("model", GREEN, "Decline highest scorecard risk"),
        ):
            s = p[p["policy"] == pol]
            y = s[col] * (100 if col == "default_rate" else 1)
            ax.plot(100 * s["approve_share"], y, marker="o", color=c, label=name, lw=2)
        ax.set_xlabel("Share of applications approved (%)")
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Approval policies on held-out cohorts FY2015-FY2020", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_policy.png", dpi=200)
    plt.close(fig)


def calibration():
    dec = pd.read_csv(TAB / "q1_deciles.csv")
    c = pd.read_csv(TAB / "q1_by_cohort.csv")
    c = c[c["model"] == "M2 scorecard"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([0, 10], [0, 10], ls="--", color=GRID)
    axes[0].plot(100 * dec["predicted"], 100 * dec["observed"], "o", color=GREEN)
    axes[0].set_xlabel("Predicted early charge-off (%)")
    axes[0].set_ylabel("Observed (%)")
    axes[0].set_title("By risk decile", fontsize=10)
    axes[1].plot(c["fy"], 100 * c["default_rate"], marker="o", color=INK, label="Observed")
    axes[1].plot(c["fy"], 100 * c["mean_predicted"], marker="o", color=GREEN, ls="--", label="Predicted")
    axes[1].annotate(
        "CARES Act payment\nrelief (FY2020)",
        xy=(2020, 100 * c["default_rate"].iloc[-1]),
        xytext=(2017.6, 2.75),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#52514e"),
    )
    axes[1].set_xlabel("Approval fiscal year")
    axes[1].set_ylabel("Early charge-off (%)")
    axes[1].set_title("By cohort", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Scorecard calibration on held-out cohorts", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_calibration.png", dpi=200)
    plt.close(fig)


def stability():
    s = json.load(open(TAB / "summary.json"))["q3"]["psi_inputs"]
    sp = json.load(open(TAB / "exploratory_spread.json"))["psi"]
    fys = sorted(s)
    feats = [
        "interest_rate",
        "guarantee_share",
        "business_age",
        "processing",
        "log_amount",
        "sector",
        "state",
    ]
    names = [
        "Interest rate",
        "Guarantee share",
        "Business age",
        "Processing method",
        "Loan amount",
        "Industry",
        "State",
    ]
    m = np.array([[s[f][x] for f in fys] for x in feats] + [[sp[f]["spread"] for f in fys]])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(np.minimum(m, 0.5), cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if m[i, j] > 0.3 else INK,
            )
    ax.set_xticks(range(len(fys)), [f"FY{f}" for f in fys])
    ax.set_yticks(range(len(names) + 1), names + ["Rate spread over prime*"])
    ax.grid(False)
    ax.set_title("Input drift vs FY2010-FY2015 loans (PSI; 0.25+ = act)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_stability.png", dpi=200)
    plt.close(fig)


def term_leak():
    a = json.load(open(TAB / "data_audit.json"))
    s = a["term_standard_share"]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    vals = [100 * s["no_default5"], 100 * s["default5"]]
    ax.barh([1, 0], vals, color=[GREEN, AMBER], height=0.55)
    for y, v in zip([1, 0], vals, strict=True):
        ax.text(v + 1, y, f"{v:.1f}%", va="center")
    ax.set_yticks([1, 0], ["Not charged off\nwithin 5 years", "Charged off\nwithin 5 years"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share with a standard term (60, 84, 120, 240 or 300 months)")
    ax.set_title("Why loan term was dropped: it is overwritten for loans that went bad")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_term_leak.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    auc_by_cohort()
    policy()
    calibration()
    stability()
    term_leak()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
