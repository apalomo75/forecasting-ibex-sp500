"""
utils_export.py
-----------------------------------
Reusable utility for exporting clean datasets
to the Power BI presentation layer.

All exported files are saved under:
../powerbi_presentation/datasets/

Example usage:
>>> from src.utils_export import export_for_powerbi
>>> export_for_powerbi(df, "var_es_results")
"""

from pathlib import Path
import pandas as pd


def export_for_powerbi(df: pd.DataFrame, name: str):
    """
    Export a DataFrame to the Power BI datasets folder.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to export.
    name : str
        Output file name (without extension).

    Output
    ------
    Saves a CSV file in: ../powerbi_presentation/datasets/{name}.csv
    """
    # Define Power BI export path (relative to notebooks/)
    path = Path("../powerbi_presentation/datasets")
    path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    file = path / f"{name}.csv"
    df.to_csv(file, index=False)

    print(f"✅ Exported for Power BI: {file}")
