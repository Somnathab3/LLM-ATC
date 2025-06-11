from pathlib import Path

import pandas as pd

from src.llm_atc.data import load_adsb_data


def test_load_adsb(tmp_path):
    f = tmp_path / "adsb.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(f, index=False)
    df = load_adsb_data(f)
    assert not df.empty
