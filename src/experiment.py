"""Run fair MNIST comparisons and write reproducible result artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import ExperimentConfig
from .models import build_mlp


HIDDEN_ONLY_SUITE: tuple[tuple[str, str, str], ...] = (
    ("sigmoid_hidden__softmax_output", "sigmoid", "softmax"),
    ("tanh_hidden__softmax_output", "tanh", "softmax"),
    ("relu_hidden__softmax_output", "relu", "softmax"),
    ("softplus_hidden__softmax_output", "softplus", "softmax"),
)

# This suite is for interpreting the textbook table, not for production use.
# A non-Softmax output is not a normalized class-probability distribution.
LEGACY_ALL_LAYERS_SUITE: tuple[tuple[str, str, str], ...] = (
    ("softmax_everywhere", "softmax", "softmax"),
    ("relu_everywhere", "relu", "relu"),
    ("softplus_everywhere", "softplus", "softplus"),
    ("tanh_everywhere", "tanh", "tanh"),
    ("sigmoid_everywhere", "sigmoid", "sigmoid"),
    ("relu_hidden__softmax_output", "relu", "softmax"),
)


def choose_suite(name: str) -> tuple[tuple[str, str, str], ...]:
    if name == "hidden":
        return HIDDEN_ONLY_SUITE
    if name == "legacy":
        return LEGACY_ALL_LAYERS_SUITE
    if name == "both":
        return HIDDEN_ONLY_SUITE + LEGACY_ALL_LAYERS_SUITE
    raise ValueError("suite must be one of: hidden, legacy, both")


def _save_history_plot(history: dict[str, list[float]], label: str, output_dir: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title=f"Loss — {label}", xlabel="Epoch", ylabel="Training objective")
    axes[1].plot(epochs, history["accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set(title=f"Accuracy — {label}", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"history_{label}.png", dpi=160)
    plt.close(fig)


def _save_summary_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("test_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ordered["experiment"], ordered["test_accuracy"], color="#2563eb")
    ax.set(title="MNIST test accuracy by activation design", ylabel="Test accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, ordered["test_accuracy"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.015, f"{score:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_accuracy.png", dpi=160)
    plt.close(fig)


def _save_prediction_examples(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    output_dir: Path,
) -> None:
    probabilities = model.predict(x_test[:16], verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = y_test[:16].argmax(axis=1)
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        axis.imshow(x_test[index].reshape(28, 28), cmap="gray")
        color = "#15803d" if predicted[index] == actual[index] else "#dc2626"
        axis.set_title(f"actual={actual[index]}, pred={predicted[index]}", color=color, fontsize=8)
        axis.axis("off")
    fig.suptitle(f"Prediction examples — {label}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / f"predictions_{label}.png", dpi=160)
    plt.close(fig)


def run_suite(config: ExperimentConfig, suite_name: str) -> pd.DataFrame:
    """Train each activation design and save CSV/PNG/JSON results."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.seed)
    x_train, y_train, x_test, y_test = load_mnist_for_experiment(config)
    results: list[dict[str, float | str]] = []

    for label, hidden_activation, output_activation in choose_suite(suite_name):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.seed)
        print(f"\n=== {label} ===")
        model = build_mlp(config, hidden_activation, output_activation)
        started = time.perf_counter()
        fitted = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=2,
        )
        elapsed = time.perf_counter() - started
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        history = {key: [float(value) for value in values] for key, values in fitted.history.items()}
        _save_history_plot(history, label, config.output_dir)
        _save_prediction_examples(model, x_test, y_test, label, config.output_dir)
        results.append(
            {
                "experiment": label,
                "hidden_activation": hidden_activation,
                "output_activation": output_activation,
                "loss": config.loss_name,
                "optimizer": config.optimizer_name,
                "train_accuracy": history["accuracy"][-1],
                "val_accuracy": history["val_accuracy"][-1],
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "seconds": elapsed,
            }
        )

    summary = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    summary.to_csv(config.output_dir / "metrics.csv", index=False)
    _save_summary_plot(summary, config.output_dir)
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_mnist_for_experiment(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Late import keeps simple static checks independent of a dataset download."""

    from .data import load_mnist

    return load_mnist(config)
