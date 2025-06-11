import pandas as pd
from src.llm_atc.model import BaselineModel


def test_baseline_train(tmp_path):
    df = pd.DataFrame({"f1": [0, 1], "conflict": [0, 1]})
    model = BaselineModel()
    acc = model.train(df)
    path = tmp_path / "model.joblib"
    model.save(path)
    assert path.exists()
