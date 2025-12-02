"""
volatility.py
-----------------------------------
Functions for volatility modeling and risk backtesting (EGARCH, GARCH, VaR).
Used in Notebook 04 — EGARCH & Volatility Backtesting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm, t


def estimate_egarch(series, dist="t", label="Series"):
    """Estimate an EGARCH(1,1) model with Student-t or Normal distribution."""
    print(f"\nEstimating EGARCH(1,1) for {label} using {dist}-distribution...")
    am = arch_model(series, mean="Constant", vol="EGARCH", p=1, q=1, dist=dist)
    model_fit = am.fit(disp="off")
    print(model_fit.summary())
    return model_fit


def compute_var(variance, dist, alpha=0.05):
    """Compute Value-at-Risk from conditional variance and innovation distribution."""
    sigma = np.sqrt(variance)
    q = dist.ppf(alpha)
    return q * sigma


def backtest_var(returns, var_series, alpha=0.05):
    """Simple VaR backtesting: Kupiec (unconditional coverage) and hit ratio."""
    exceptions = (returns < var_series).astype(int)
    n = len(exceptions)
    x = exceptions.sum()
    pi_hat = x / n

    if x in [0, n]:
        lr_uc = np.nan
        p_value = np.nan
    else:
        lr_uc = -2 * (
            np.log(((1 - alpha) ** (n - x)) * (alpha ** x))
            - np.log(((1 - pi_hat) ** (n - x)) * (pi_hat ** x))
        )
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_uc, 1)

    return {
        "alpha": alpha,
        "n": n,
        "violations": int(x),
        "violation_ratio": round(pi_hat / alpha, 2),
        "kupiec_lr": lr_uc,
        "p_value": p_value,
    }


def plot_volatility(series, cond_vol, label, fig_path=None):
    """
    Plot conditional volatility (sigma_t) from an estimated GARCH/EGARCH model.

    Parameters
    ----------
    series : pd.Series
        Original return series.
    cond_vol : pd.Series
        Conditional volatility from fitted model (model_fit.conditional_volatility).
    label : str
        Series label (e.g. 'IBEX' or 'SPX').
    fig_path : Path or str, optional
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, cond_vol, color="steelblue", linewidth=1.2)
    plt.title(f"Conditional Volatility — {label}", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("σₜ (Conditional Volatility)")
    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved → {fig_path}")
    plt.show()
