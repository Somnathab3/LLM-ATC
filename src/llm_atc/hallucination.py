from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Envelope:
    lower: pd.Series
    upper: pd.Series

    @classmethod
    def from_training(cls, df: pd.DataFrame) -> "Envelope":
        mean = df.mean()
        std = df.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        return cls(lower=lower, upper=upper)

    def check(self, sample: pd.Series) -> Dict[str, float]:
        out_of_bounds = {}
        for col in sample.index:
            val = sample[col]
            if val < self.lower[col] or val > self.upper[col]:
                dist = min(self.lower[col] - val, val - self.upper[col])
                out_of_bounds[col] = abs(dist)
        return out_of_bounds


def severity_score(out_of_bounds: Dict[str, float]) -> float:
    if not out_of_bounds:
        return 0.0
    return float(np.mean(list(out_of_bounds.values())))
