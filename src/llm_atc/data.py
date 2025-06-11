import json
from pathlib import Path
from typing import Tuple

import pandas as pd


def load_adsb_data(path: Path) -> pd.DataFrame:
    """Load ADS-B data from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def load_scat_data(path: Path) -> pd.DataFrame:
    """Load SCAT dataset CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Simple conflict labeling based on separation minima."""
    df = df.copy()
    df["conflict"] = (df.get("horizontal_sep", 10000) < 5) & (
        df.get("vertical_sep", 2000) < 1000
    )
    return df


def save_json_trace(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
