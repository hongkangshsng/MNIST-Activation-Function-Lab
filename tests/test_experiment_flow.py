from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src import experiment


class _FakeModel:
    def __init__(self) -> None:
        self.validation_data = None
        self.evaluated_data = None

    def fit(self, x, y, *, validation_data, **kwargs):
        self.validation_data = validation_data
        return type(
            "History",
            (),
            {"history": {"loss": [0.8], "accuracy": [0.7], "val_loss": [0.9], "val_accuracy": [0.6]}},
        )()

    def evaluate(self, x, y, **kwargs):
        self.evaluated_data = (x, y)
        return 0.95, 0.55


def test_run_suite_uses_validation_data_and_separate_test_data(monkeypatch, tmp_path: Path) -> None:
    x_fit = np.full((2, 784), 1.0)
    y_fit = np.eye(10)[:2]
    x_val = np.full((2, 784), 2.0)
    y_val = np.eye(10)[2:4]
    x_test = np.full((2, 784), 3.0)
    y_test = np.eye(10)[4:6]
    model = _FakeModel()

    monkeypatch.setattr(
        experiment,
        "load_mnist_for_experiment",
        lambda config: (x_fit, y_fit, x_val, y_val, x_test, y_test),
    )
    monkeypatch.setattr(experiment, "build_mlp", lambda *args: model)
    monkeypatch.setattr(experiment, "_save_history_plot", lambda *args: None)
    monkeypatch.setattr(experiment, "_save_prediction_examples", lambda *args: None)
    monkeypatch.setattr(experiment, "_save_summary_plot", lambda *args: None)
    monkeypatch.setattr(experiment, "choose_suite", lambda name: (("controlled", "relu", "softmax"),))

    summary = experiment.run_suite(ExperimentConfig(output_dir=tmp_path), "hidden")

    assert model.validation_data[0] is x_val
    assert model.validation_data[1] is y_val
    assert model.evaluated_data[0] is x_test
    assert model.evaluated_data[1] is y_test
    assert summary.loc[0, "validation_accuracy"] == 0.6
    assert summary.loc[0, "test_accuracy"] == 0.55
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "run_config.json").exists()
