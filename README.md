# LLM-ATC

A toy pipeline for ML-based conflict detection and hallucination monitoring in air-traffic control.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Prepare some `data/adsb.csv` and `data/scat.csv` files.

Run the pipeline:

```bash
python run_pipeline.py prepare-data
python run_pipeline.py train
python run_pipeline.py simulate
```

Logs go to `logs/`, reports to `reports/`.

## Tests

```bash
pytest
```
