from pathlib import Path
from typing import Iterable

import pandas as pd

from .hallucination import Envelope, severity_score
from .model import BaselineModel


class DummyBlueSky:
    """Placeholder for BlueSky simulation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> Iterable[pd.Series]:
        for _, row in self.df.iterrows():
            yield row


def simulate(
    model: BaselineModel,
    env: DummyBlueSky,
    envelope: Envelope,
    trace_path: Path,
) -> None:
    logs = []
    for step in env.run():
        prob = model.predict(step.to_frame().T).iloc[0]
        bounds = envelope.check(step)
        sev = severity_score(bounds)
        logs.append(
            {
                "step": step.to_dict(),
                "prob": float(prob),
                "hallucination": bounds,
                "severity": sev,
            }
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as f:
        import json

        json.dump(logs, f, indent=2)
