"""Optional PyTorch sequence models, including the DECODE-style architecture."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from campus_forecast.exceptions import OptionalDependencyError
from campus_forecast.models.baselines import DirectForecaster


class _TorchSequenceRegressor(DirectForecaster):
    def __init__(
        self,
        *,
        horizon_steps: int | None = None,
        hidden_size: int = 32,
        cell: str = "lstm",
        dense_units: int = 5,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 5,
        seed: int = 42,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise OptionalDependencyError(
                "PyTorch is required for sequence models. Install the research extra."
            ) from exc
        if cell not in {"lstm", "gru"}:
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.torch = torch
        self.nn = nn
        self.horizon_steps = horizon_steps
        self.hidden_size = hidden_size
        self.cell = cell
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.history: list[dict[str, float]] = []

    def _build(self, input_size: int, horizon_steps: int):
        nn = self.nn
        cell = nn.LSTM if self.cell == "lstm" else nn.GRU

        class SequenceNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.recurrent = cell(input_size, self_outer.hidden_size, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(self_outer.hidden_size, self_outer.dense_units),
                    nn.ReLU(),
                    nn.Linear(self_outer.dense_units, self_outer.dense_units),
                    nn.ReLU(),
                    nn.Linear(self_outer.dense_units, horizon_steps),
                )

            def forward(self, values):
                output, _ = self.recurrent(values)
                return self.head(output[:, -1, :])

        self_outer = self
        # Nested class resolves self_outer when instantiated after assignment.
        return SequenceNetwork()

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> _TorchSequenceRegressor:
        torch = self.torch
        torch.manual_seed(self.seed)
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 3 or y.ndim != 2 or len(x) != len(y):
            raise ValueError(
                "sequence features must be [samples, steps, variables] and targets [samples, horizons]"
            )
        self.horizon_steps = y.shape[1]
        self.model = self._build(x.shape[2], self.horizon_steps)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        validation_tensors = None
        if validation is not None:
            validation_tensors = (
                torch.from_numpy(np.asarray(validation[0], dtype=np.float32)),
                torch.from_numpy(np.asarray(validation[1], dtype=np.float32)),
            )
        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history = []
        for epoch in range(self.epochs):
            self.model.train()
            losses: list[float] = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses)) if losses else float("nan")
            if validation_tensors is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(
                        loss_fn(self.model(validation_tensors[0]), validation_tensors[1]).cpu()
                    )
            else:
                val_loss = train_loss
            self.history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss}
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Sequence model has not been fitted")
        tensor = self.torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.model.eval()
        with self.torch.no_grad():
            return self.model(tensor).cpu().numpy()


class DecodeLSTMRegressor(_TorchSequenceRegressor):
    """DECODE-style one-layer 32-unit LSTM with two five-unit dense layers."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("dense_units", 5)
        kwargs.setdefault("cell", "lstm")
        kwargs.setdefault("epochs", 20)
        kwargs.setdefault("batch_size", 64)
        super().__init__(**kwargs)


class GRURegressor(_TorchSequenceRegressor):
    """GRU comparison model with the same direct-output interface."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("dense_units", 5)
        kwargs.setdefault("cell", "gru")
        super().__init__(**kwargs)
