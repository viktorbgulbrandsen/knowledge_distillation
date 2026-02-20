from __future__ import annotations
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TeacherConfig:
    hidden_layers: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: int = 128
    n_classes: int = 6


class TeacherNN(nn.Module):
    def __init__(self, input_dim: int, cfg: TeacherConfig):
        super().__init__()

        layers = []
        prev = input_dim
        for h in cfg.hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.classifier(feats)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class TeacherTrainer:
    def __init__(
        self,
        cfg: TeacherConfig,
        patience: int = 15,
        max_epochs: int = 200,
        verbose: bool = True,
        print_every: int = 10,
    ):
        self.cfg = cfg
        self.patience = patience
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.print_every = print_every
        self.model: TeacherNN | None = None
        self.val_metric: float | None = None
        self.history: dict | None = None

    def _build(self, input_dim: int):
        model = TeacherNN(input_dim, self.cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, criterion

    def _train_epoch(self, model, optimizer, criterion, X, y):
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    def _logits(self, X, model=None):
        model = model or self.model
        if model is None:
            raise RuntimeError("Model not trained yet")

        model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return model(X).numpy()

    def predict_proba(self, X, model=None):
        logits = self._logits(X, model)
        return torch.softmax(torch.tensor(logits), dim=1).numpy()

    def predict_logits(self, X):
        return self._logits(X)

    def extract_features(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return self.model.extract_features(X).numpy()

    def fit(self, X_train, y_train, X_val, y_val, metric_fn):
        if len(np.unique(y_train)) != self.cfg.n_classes and self.verbose:
            print("Warning: some classes missing in training split")

        y_train0 = y_train - 1

        model, optimizer, criterion = self._build(X_train.shape[1])
        best_state, best_metric, best_epoch, wait = None, -1.0, 0, 0

        epoch_losses = []
        epoch_val_kappas = []

        for epoch in range(self.max_epochs):
            loss = self._train_epoch(model, optimizer, criterion, X_train, y_train0)

            probs = self.predict_proba(X_val, model)
            preds = probs.argmax(1) + 1
            metric = metric_fn(y_val, preds)

            epoch_losses.append(loss)
            epoch_val_kappas.append(metric)

            if self.verbose and self.print_every and ((epoch + 1) % self.print_every == 0):
                print(f"  epoch {epoch+1:>3d}: loss={loss:.4f}, val_κ={metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"  early stop epoch {epoch+1}, best κ={best_metric:.4f}")
                    break

        model.load_state_dict(best_state)
        self.model = model
        self.val_metric = best_metric
        self.history = {
            "losses": epoch_losses,
            "val_kappas": epoch_val_kappas,
            "best_epoch": best_epoch,
            "best_val_kappa": best_metric,
        }
        return self