"""
visualization.py
Reusable plotting functions for exploratory data analysis.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_daily_returns(returns, save_path=None):
    """
    Plot daily log returns for IBEX and S&P 500.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame containing IBEX_ret and SPX_ret columns.
    save_path : str or Path, optional
        If provided, saves the figure to the specified path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    returns.plot(ax=ax, linewidth=0.8)
    ax.set_title("Daily Log Returns — IBEX vs S&P 500 (2000–present)")
    ax.set_ylabel("Log return")
    ax.legend(["IBEX", "S&P 500"])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_price_trajectories(ibex, spx, save_path=None):
    """Plot price trajectories of IBEX and S&P 500."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ibex.index, ibex["IBEX"], label="IBEX 35", alpha=0.9)
    ax.plot(spx.index, spx["SPX"], label="S&P 500", alpha=0.9)
    ax.set_title("Price Trajectories — IBEX 35 vs S&P 500 (2000–present)")
    ax.set_ylabel("Index level")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_rolling_volatility(returns, window=20, save_path=None):
    """Plot 20-day rolling annualized volatility for both indices."""
    rolling_vol = returns.rolling(window).std() * (252 ** 0.5)

    fig, ax = plt.subplots(figsize=(12, 5))
    rolling_vol.plot(ax=ax, linewidth=1.0)
    ax.set_title(f"Rolling {window}-Day Annualized Volatility")
    ax.set_ylabel("Volatility (%)")
    ax.legend(["IBEX", "S&P 500"])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# ============================================================
# Additional plots for ARIMA / Prophet modules
# ============================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def plot_acf_pacf(series, lags=40, title="", save_path=None):
    """
    Plot ACF and PACF for a given time series.
    Returns the matplotlib Figure object so it can be displayed in Jupyter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

 
    return fig


def plot_forecast(forecast_df, title="", save_path=None):
    """
    Plot forecasted values by horizon and return the Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(forecast_df["Horizon"], forecast_df["Forecast"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("Predicted return")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


    return fig

    # ============================================================
# General plot style for consistency across all notebooks
# ============================================================

import seaborn as sns

def set_plot_style():
    """
    Apply a unified aesthetic style for all project visualizations.
    Used in EDA, EGARCH, VaR/ES, and ML modules.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.4,
        "font.family": "DejaVu Sans",
    })
    print("Plot style applied (whitegrid, consistent font and sizing).")




