"""Layer 3 output: writing final tables to results/ (metrics, explainability, diagnostics, predictions)."""
import pandas as pd

from ..setup import config


def save_table(df, name, decimals=4, index=False, folder=None):
    """Round numeric values (if decimals is not None) and write to results/metrics/<name>.csv
    (or another folder). Returns the file path."""
    folder = folder or config.METRIC_DIR
    folder.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if decimals is not None:
        if isinstance(out, pd.Series):
            out = out.round(decimals)
        else:
            num = out.select_dtypes("number").columns
            out[num] = out[num].round(decimals)
    path = folder / f"{name}.csv"
    out.to_csv(path, index=index)
    return path
