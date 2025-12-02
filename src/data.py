"""
data.py
Utility functions for loading and managing financial time series data.
"""

import pandas as pd
from pathlib import Path


def load_clean_data():
    """
    Load cleaned IBEX and S&P 500 datasets from /data/interim/.
    Returns
    -------
    ibex : pd.DataFrame
        Cleaned IBEX 35 index with Date as index.
    spx : pd.DataFrame
        Cleaned S&P 500 index with Date as index.
    """
    data_path = Path("../data/interim")

    ibex = pd.read_csv(data_path / "IBEX_clean.csv", index_col=0, parse_dates=True)
    spx = pd.read_csv(data_path / "SPX_clean.csv", index_col=0, parse_dates=True)

    return ibex, spx
