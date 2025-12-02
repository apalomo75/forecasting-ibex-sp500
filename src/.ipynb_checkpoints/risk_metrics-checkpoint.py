"""
risk_metrics.py
-----------------------------------
Risk measurement utilities for VaR, Expected Shortfall (ES),
and backtesting (Christoffersen independence test + plotting).

Used in Notebook 05 — VaR & Expected Shortfall.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, chi2


def compute_es(sigma, dof, alpha=0.01):
    """
    Compute Expected Shortfall (ES) for a left-tail t-Student distribution.

    Parameters
    ----------
    sigma : pd.Series or np.ndarray
        Conditional volatility series (same scale as returns used for VaR).
    dof : float
        Degrees of freedom of the Student-t distribution (nu from EGARCH model).
    alpha : float, default 0.01
        Tail probability (e.g. 0.01 for ES at 99% confidence).

    Returns
    -------
    es_series : pd.Series
        Expected Shortfall series (negative values for losses).
    """
    sigma = pd.Series(sigma)
    q = student_t.ppf(alpha, dof)  # negative for alpha < 0.5
    f_q = student_t.pdf(q, dof)

    # ES of standard t(0,1,nu) in the left tail: E[X | X <= q_alpha]
    es_std = -f_q * (dof + q**2) / ((dof - 1) * alpha)

    es = es_std * sigma
    es.name = f"ES_{int((1-alpha)*100)}"
    return es


def christoffersen_test(exceedances):
    """
    Christoffersen (1998) test for independence of VaR exceedances.

    Parameters
    ----------
    exceedances : array-like
        Binary sequence where 1 indicates a VaR exceedance (loss > VaR),
        and 0 otherwise.

    Returns
    -------
    results : dict
        Dictionary with transition counts, LR statistic and p-value.
    """
    x = pd.Series(exceedances).astype(int).dropna()

    if len(x) < 2:
        return {
            "n00": np.nan,
            "n01": np.nan,
            "n10": np.nan,
            "n11": np.nan,
            "lr_ind": np.nan,
            "p_value": np.nan,
        }

    # Transitions I_t -> I_{t+1}
    x_t = x[:-1].values
    x_tp1 = x[1:].values

    n00 = np.sum((x_t == 0) & (x_tp1 == 0))
    n01 = np.sum((x_t == 0) & (x_tp1 == 1))
    n10 = np.sum((x_t == 1) & (x_tp1 == 0))
    n11 = np.sum((x_t == 1) & (x_tp1 == 1))

    # Handle edge cases where some transitions do not occur
    if (n01 + n00) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "lr_ind": np.nan,
            "p_value": np.nan,
        }

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Log-likelihoods
    logL0 = (
        (n00 + n01) * np.log(1 - pi)
        + (n10 + n11) * np.log(pi)
    )

    logL1 = (
        n00 * np.log(1 - pi_01)
        + n01 * np.log(pi_01)
        + n10 * np.log(1 - pi_11)
        + n11 * np.log(pi_11)
    )

    lr_ind = -2 * (logL0 - logL1)
    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
        "lr_ind": lr_ind,
        "p_value": p_value,
    }


def plot_var_es(returns, var95, var99, es99, label="Index", fig_path=None):
    """
    Plot daily returns together with VaR95, VaR99 and ES99.

    Parameters
    ----------
    returns : pd.Series
        Return series (same scale as VaR and ES).
    var95 : pd.Series
        VaR at 95% confidence (alpha = 0.05).
    var99 : pd.Series
        VaR at 99% confidence (alpha = 0.01).
    es99 : pd.Series
        Expected Shortfall at 99% confidence.
    label : str
        Name of the index (e.g. 'IBEX', 'SPX').
    fig_path : Path or str, optional
        If provided, the figure is saved to this path.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(returns.index, returns.values, label="Returns", linewidth=0.8)
    plt.plot(var95.index, var95.values, label="VaR 95%", linestyle="--", linewidth=1.0)
    plt.plot(var99.index, var99.values, label="VaR 99%", linestyle="--", linewidth=1.0)
    plt.plot(es99.index, es99.values, label="ES 99%", linestyle="-.", linewidth=1.0)

    plt.title(f"Returns vs VaR/ES — {label}", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Return (same scale as input)")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved → {fig_path}")

    plt.show()
