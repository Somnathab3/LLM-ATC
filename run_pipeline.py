import argparse
from pathlib import Path

import pandas as pd

from src.llm_atc.data import (
    load_adsb_data,
    load_scat_data,
    label_conflicts,
    save_json_trace,
)
from src.llm_atc.hallucination import Envelope
from src.llm_atc.model import BaselineModel
from src.llm_atc.simulate import DummyBlueSky, simulate


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"


def prepare_data() -> None:
    adsb_path = DATA_DIR / "adsb.csv"
    scat_path = DATA_DIR / "scat.csv"
    adsb = load_adsb_data(adsb_path)
    scat = load_scat_data(scat_path)
    df = pd.concat([adsb, scat], ignore_index=True, sort=False)
    df = label_conflicts(df)
    df.to_csv(DATA_DIR / "prepared.csv", index=False)


def train() -> None:
    df = pd.read_csv(DATA_DIR / "prepared.csv")
    model = BaselineModel()
    acc = model.train(df)
    model.save(MODEL_DIR / "baseline.joblib")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "train_metrics.txt", "w") as f:
        f.write(f"accuracy: {acc}\n")


def simulate_mode() -> None:
    df = pd.read_csv(DATA_DIR / "prepared.csv")
    model = BaselineModel()
    model.load(MODEL_DIR / "baseline.joblib")
    envelope = Envelope.from_training(df.drop(columns=["conflict"]))
    env = DummyBlueSky(df.drop(columns=["conflict"]))
    simulate(model, env, envelope, LOG_DIR / "trace.json")


def evaluate() -> None:
    print("Evaluation placeholder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare-data", "train", "simulate", "evaluate"]
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    if args.mode == "prepare-data":
        prepare_data()
    elif args.mode == "train":
        train()
    elif args.mode == "simulate":
        simulate_mode()
    elif args.mode == "evaluate":
        evaluate()
