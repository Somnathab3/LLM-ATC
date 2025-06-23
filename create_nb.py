import json
cells = []

def md(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': text
    }

def code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': text
    }

cells.append(md('# Simulation and Quantification of ML-Based Hallucination Effects on Safety Margins in En-route Air Traffic Control'))
cells.append(md('This notebook provides a proof-of-concept implementation for the thesis **"Simulation and Quantification of ML-Based Hallucination Effects on Safety Margins in En-route ATC"**. It demonstrates data ingestion, feature engineering, a lightweight LLM-based conflict detector, hallucination detection, and integration with the BlueSky-Gym simulator.'))

cells.append(md('## 1. Environment Setup & Dependencies\nInstall the required libraries. The key packages include:\n- **pandas, numpy, matplotlib, scikit-learn**: data processing and traditional ML helpers.\n- **ollama**: running and fine-tuning lightweight LLMs locally.\n- **unsloth**: simple quantization and PEFT helpers for Llama/Mistral.\n- **bitsandbytes, accelerate**: memory efficient 4-bit quantized training.'))

cells.append(code('!pip install pandas numpy matplotlib scikit-learn ollama unsloth bitsandbytes accelerate'))

cells.append(md('## 2. SCAT Dataset Ingestion & Parsing\nDownload the 13-week SCAT JSON archives from Mendeley and unpack them. Each archive contains flight plans, radar plots, predicted trajectories, weather, and airspace information.'))

cells.append(code("""import pandas as pd
from pathlib import Path
import json

data_path = Path('data/scat')  # update with the actual path after download

# Example parser for a flight plan JSON file
fpl_base_files = list(data_path.glob('**/fpl_base.json'))
all_fpl_base = []
for fp in fpl_base_files:
    with open(fp) as f:
        all_fpl_base.extend(json.load(f))

fpl_base_df = pd.DataFrame(all_fpl_base)
fpl_base_df.head()
"""))

cells.append(md('Similar parsers can be written for the remaining JSON files such as `fpl_clearance.json`, `I062/105` radar plots (positions & altitudes), `predicted_trajectory.json`, `grib_meteo.json`, and `airspace.json`. The resulting DataFrames provide structured access to the SCAT dataset.'))

cells.append(md('## 3. Feature Engineering & Conflict Labels\nWe first normalize timestamps and filter to 5-minute windows. For each pair of flights at a given time step we compute horizontal and vertical separation, expressed in nautical miles (NM) and feet. We also compute the time-to-closest-approach (TCA). If the predicted separation within the next five minutes is below the ICAO minima of **5 NM** horizontally or **1000 ft** vertically, the pair is labeled as a conflict.'))

cells.append(code("""import numpy as np

def separation_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # Earth radius in nautical miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Example placeholder feature computation
# df contains positions of two aircraft at time t
# features = compute_features(df)
# X, y = build_dataset(features)
"""))

cells.append(md('## 4. Lightweight LLM Conflict-Detector with Ollama + Unsloth\nWe use Ollama to run a small Llama-2-7B or Mistral-7B model. The model is quantized to 4-bit using Unsloth for minimal resource usage. Fine-tuning is carried out via LoRA/QLoRA.'))

cells.append(code("""import torch
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model

model, tokenizer = FastLanguageModel.from_pretrained('mistral:latest', load_in_4bit=True)
config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(model, config)

# Training loop placeholder
# for batch in train_loader:
#     loss = model(**batch).loss
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
"""))

cells.append(code("""# Example evaluation placeholder
# predictions = model.predict(test_features)
# print(classification_report(test_labels, predictions))
"""))

cells.append(md('## 5. Hallucination Detection Envelope\nWe compute empirical feature bounds (mean ± 3σ) from the training data. Inputs outside these bounds are considered out-of-distribution (OOD). Whenever the LLM predicts a label for an OOD input, a hallucination flag is recorded.'))

cells.append(code("""feature_means = X_train.mean(axis=0)
feature_stds = X_train.std(axis=0)
low_bounds = feature_means - 3 * feature_stds
high_bounds = feature_means + 3 * feature_stds

def is_ood(sample):
    return ((sample < low_bounds) | (sample > high_bounds)).any()
"""))

cells.append(md('## 6. BlueSky Integration for End-to-End CD&R\nWe integrate the conflict detector with the BlueSky-Gym simulator. The LLM issues conflict alerts two minutes ahead and proposes resolutions ("Climb +1000 ft" or "Turn ±20°").'))

cells.append(code("""# import gym
# import bluesky_gym
# env = bluesky_gym.init('small_sector')
#
# for step in range(max_steps):
#     obs = env.get_state()
#     conflict = model.predict(obs)
#     if conflict:
#         env.apply_clearance('CLIMB', 1000)
#     env.step()
"""))

cells.append(md('## 7. Metrics & Visualization'))

cells.append(code("""import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Placeholder for metrics computation
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(cm)
# disp.plot()
# plt.show()
"""))

cells.append(md('## 8. Distribution Shift Analysis\nWe systematically vary traffic density and measure hallucination rate, safety margin erosion, and reactive workload.'))

cells.append(code("""# for density in [1.0, 1.5, 2.0]:
#     results = run_simulation(density=density)
#     plot_metrics(results)
"""))

cells.append(md('## 9. Traceability & Audit Logs\nWe save detailed logs linking raw input, model version, prompts, predicted risk, issued resolutions, and simulation outcomes.'))

cells.append(code("""import csv

with open('audit_log.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'model_version', 'prompt', 'prediction', 'resolution', 'outcome', 'hallucination'])
    # writer.writerow([...])
"""))

cells.append(md('## 10. Documentation & Reproducibility\nRun this notebook on a GPU-equipped machine with the required libraries installed. Follow the instructions in each section to reproduce the data processing, model training, and simulation steps. Training data (SCAT) remains private; only the code and the trained model weights are shared here.'))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'pygments_lexer': 'ipython3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('LLM_ATC_demo.ipynb', 'w') as f:
    json.dump(nb, f)
