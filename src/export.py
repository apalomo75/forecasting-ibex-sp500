# export.py — unified export pipeline for Power BI

from pathlib import Path

# Windows-accessible export folder
EXPORT_DIR = Path("/mnt/c/Users/apalo/Desktop/forescasting_results")

# Ensure folder exists
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def export(df, name):
    """
    Export a DataFrame to the Power BI folder (Windows).
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        name (str): File name without extension.
    """
    filepath = EXPORT_DIR / f"{name}.csv"
    df.to_csv(filepath, index=False)
    print(f"[EXPORT] {name}.csv saved to {filepath}")
