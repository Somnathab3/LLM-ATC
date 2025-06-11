from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class BaselineModel:
    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def train(self, df: pd.DataFrame, target: str = "conflict") -> float:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(df)[:, 1], index=df.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)


class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=4, dim_feedforward=hidden_dim
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        out = self.fc(x.mean(dim=0))
        return torch.sigmoid(out)


class TimeSeriesModel:
    def __init__(self, input_dim: int):
        self.net = TransformerModel(input_dim)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(
        self, tensor: torch.Tensor, labels: torch.Tensor, epochs: int = 5
    ) -> float:
        self.net.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            out = self.net(tensor)
            loss = self.criterion(out.squeeze(), labels.float())
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.net(tensor).squeeze()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path, input_dim: int) -> None:
        self.net = TransformerModel(input_dim)
        self.net.load_state_dict(torch.load(path))
