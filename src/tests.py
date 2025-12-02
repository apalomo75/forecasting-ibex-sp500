"""
tests.py
Statistical diagnostics for financial time series.
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera


def adf_test(series, name="Series"):
    """Perform Augmented Dickey–Fuller test for stationarity."""
    result = adfuller(series, autolag="AIC")
    print(f"--- ADF Test: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    print("------------------------\n")
    return result[0], result[1]


def arch_lm_test(series, name="Series", lags=12):
    """Perform Engle’s ARCH–LM test for conditional heteroskedasticity."""
    lm_stat, lm_pval, f_stat, f_pval = het_arch(series, nlags=lags)
    print(f"--- ARCH–LM Test: {name} ---")
    print(f"LM Statistic: {lm_stat:.4f}")
    print(f"p-value (LM): {lm_pval:.4f}")
    print(f"F Statistic:  {f_stat:.4f}")
    print(f"p-value (F):  {f_pval:.4f}")
    print("-----------------------------\n")
    return lm_stat, lm_pval


def jb_test(series, name="Series"):
    """Perform Jarque–Bera test for normality."""
    jb_stat, jb_pval = jarque_bera(series)
    print(f"--- Jarque–Bera Test: {name} ---")
    print(f"JB Statistic: {jb_stat:.4f}")
    print(f"p-value:      {jb_pval:.4f}")
    print("-------------------------------\n")
    return jb_stat, jb_pval
