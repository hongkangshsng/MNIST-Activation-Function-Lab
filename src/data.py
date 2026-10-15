"""MNIST loading and preprocessing used by every experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import ExperimentConfig


def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
    train_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic, disjoint train/validation split.

    The validation examples come only from MNIST's official training split.
    This prevents the official test split from influencing activation selection
    or hyperparameter decisions.
    """

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must contain the same number of samples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    indices = np.random.default_rng(seed).permutation(len(x_train))
    if train_limit is not None:
        if train_limit < 2:
            raise ValueError("train_limit must be at least 2")
        indices = indices[: min(train_limit, len(indices))]

    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    if validation_count >= len(indices):
        raise ValueError("validation split leaves no training samples")
    validation_indices = indices[:validation_count]
    fitting_indices = indices[validation_count:]
    return (
        x_train[fitting_indices],
        y_train[fitting_indices],
        x_train[validation_indices],
        y_train[validation_indices],
    )


def load_mnist(
    config: "ExperimentConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train/validation data plus an untouched test set.

    An original MNIST image is 28 x 28, with a grayscale value in [0, 255].
    A Dense layer expects one feature vector per sample, so 28 x 28 becomes
    784.  Dividing by 255 maps every feature to [0, 1], making gradient-based
    optimisation much more stable.  Labels 0--9 become 10-element one-hot
    vectors because this lab uses categorical cross-entropy.
    """

    # Import lazily so deterministic split tests do not need TensorFlow or a
    # network download.
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, config.input_features).astype("float32") / 255.0
    x_test = x_test.reshape(-1, config.input_features).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, config.classes).astype("float32")
    y_test = tf.keras.utils.to_categorical(y_test, config.classes).astype("float32")

    if config.test_limit is not None:
        x_test, y_test = x_test[: config.test_limit], y_test[: config.test_limit]
    x_fit, y_fit, x_val, y_val = split_train_validation(
        x_train,
        y_train,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        train_limit=config.train_limit,
    )
    return x_fit, y_fit, x_val, y_val, x_test, y_test
