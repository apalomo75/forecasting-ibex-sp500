"""
build_technical_indicators.py
------------------------------------------------
Generate RSI, MACD, Momentum, and Volatility features
for IBEX 35 and S&P 500 returns series.

Output:
    ../data/features/technical_indicators.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Utility functions
# -----------------------------
def compute_RSI(series, window=14):
    """Compute RSI using Wilder's method with correct index alignment."""
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))

    return RSI


def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram


def compute_momentum(series, window=5):
    return series - series.shift(window)


def compute_volatility(series, window=20):
    return series.rolling(window).std() * np.sqrt(252)


# -----------------------------
# Main processing function
# -----------------------------
def build_indicators():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "features"
    out_path = data_dir / "technical_indicators.csv"

    # Load returns
    df = pd.read_csv(data_dir / "returns.csv", index_col=0, parse_dates=True)
    df = df.rename(columns={"IBEX_ret": "IBEX", "SPX_ret": "SPX"})

    # Output dataframe
    df_ind = pd.DataFrame(index=df.index)

    # Compute indicators for both indices
    for col in ["IBEX", "SPX"]:
        df_ind[f"RSI_{col}"] = compute_RSI(df[col])
        df_ind[f"MACD_{col}"] = compute_MACD(df[col])
        df_ind[f"Momentum_{col}"] = compute_momentum(df[col])
        df_ind[f"Volatility_{col}"] = compute_volatility(df[col])

    # Save result
    df_ind.to_csv(out_path)
    print(f"✅ Technical indicators saved → {out_path}")
    print(df_ind.tail())


if __name__ == "__main__":
    build_indicators()
