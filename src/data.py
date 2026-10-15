"""MNIST loading and preprocessing used by every experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from .config import ExperimentConfig


def load_mnist(config: "ExperimentConfig") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened, normalized images and one-hot labels.

    An original MNIST image is 28 x 28, with a grayscale value in [0, 255].
    A Dense layer expects one feature vector per sample, so 28 x 28 becomes
    784.  Dividing by 255 maps every feature to [0, 1], making gradient-based
    optimisation much more stable.  Labels 0--9 become 10-element one-hot
    vectors because this lab uses categorical cross-entropy.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, config.input_features).astype("float32") / 255.0
    x_test = x_test.reshape(-1, config.input_features).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, config.classes).astype("float32")
    y_test = tf.keras.utils.to_categorical(y_test, config.classes).astype("float32")

    if config.train_limit is not None:
        x_train, y_train = x_train[: config.train_limit], y_train[: config.train_limit]
    if config.test_limit is not None:
        x_test, y_test = x_test[: config.test_limit], y_test[: config.test_limit]
    return x_train, y_train, x_test, y_test
