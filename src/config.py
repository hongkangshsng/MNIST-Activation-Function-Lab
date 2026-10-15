"""Experiment configuration and reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_ACTIVATIONS = ("sigmoid", "tanh", "relu", "softplus", "softmax")
SUPPORTED_LOSSES = ("crossentropy", "mse")
SUPPORTED_OPTIMIZERS = ("adam", "sgd")


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters kept constant while activation functions are compared."""

    epochs: int = 12
    batch_size: int = 128
    hidden_units: int = 256
    # None selects a sensible framework-style default: Adam=0.001, SGD=0.01.
    learning_rate: float | None = None
    loss_name: str = "crossentropy"
    optimizer_name: str = "adam"
    train_limit: int | None = None
    test_limit: int | None = None
    seed: int = 42
    output_dir: Path = Path("results")

    @property
    def input_features(self) -> int:
        return 28 * 28

    @property
    def classes(self) -> int:
        return 10
