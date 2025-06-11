import pandas as pd
from src.llm_atc.hallucination import Envelope


def test_envelope():
    df = pd.DataFrame({"a": [1, 2, 3]})
    env = Envelope.from_training(df)
    sample = pd.Series({"a": 100})
    res = env.check(sample)
    assert "a" in res
