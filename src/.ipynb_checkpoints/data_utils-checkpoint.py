from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Robust project paths (independent from working directory)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DATA_DIR = DATA_DIR / "models"


# =========================================================
# LOADERS
# =========================================================

def load_returns_data():
    """
    Load returns.csv ensuring Date column is properly detected.
    Works even if pandas adds 'Unnamed: 0'.
    """
    path = FEATURES_DIR / "returns.csv"
    df = pd.read_csv(path)

    # Detect date column dynamically
    date_col = None
    for c in ["Date", "date", "Unnamed: 0"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            f"No valid date column found in returns.csv. Columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df


def load_technical_indicators(suffix: str | None = None):
    """
    Load RSI, MACD, Momentum, Volatility indicators.
    Handles both 'Date' and 'Unnamed: 0'.
    """
    path = FEATURES_DIR / "technical_indicators.csv"
    df = pd.read_csv(path)

    # Detect index column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    return df.add_suffix(suffix) if suffix else df


def load_egarch_vol(index_code: str):
    """Load EGARCH conditional volatility."""
    fname = f"egarch_{index_code.lower()}_vol.csv"
    return pd.read_csv(MODELS_DATA_DIR / fname, parse_dates=["Date"], index_col="Date")


# =========================================================
# FEATURE ENGINEERING HELPERS
# =========================================================

def create_lagged_features(df: pd.DataFrame, columns: list[str], lags: list[int] = [1, 2, 3]):
    """Generate lagged versions of input columns for ML/TS models."""
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_direction_label(df: pd.DataFrame, column: str = "return"):
    """
    Add binary direction label (1 if next day's return is positive).
    """
    out = df.copy()
    out["direction"] = (out[column].shift(-1) > 0).astype(int)
    return out.dropna()
