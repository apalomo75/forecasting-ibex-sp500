"""
src package
Core functions for the Financial Forecasting project.
Includes:
- Data loading and preprocessing utilities
- Visualization helpers for EDA
- Statistical testing functions
"""

from .data import load_clean_data
from .visualization import (
    plot_daily_returns,
    plot_price_trajectories,
    plot_rolling_volatility,
)
from .tests import adf_test, arch_lm_test, jb_test

__all__ = [
    "load_clean_data",
    "plot_daily_returns",
    "plot_price_trajectories",
    "plot_rolling_volatility",
    "adf_test",
    "arch_lm_test",
    "jb_test",
]
