"""Model factory for controlled activation-function experiments."""

from __future__ import annotations

import tensorflow as tf

from .config import (
    ExperimentConfig,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_LOSSES,
    SUPPORTED_OPTIMIZERS,
)


def _make_optimizer(config: ExperimentConfig) -> tf.keras.optimizers.Optimizer:
    if config.optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=config.learning_rate or 1e-3)
    if config.optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=config.learning_rate or 1e-2)
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def _make_loss(config: ExperimentConfig) -> tf.keras.losses.Loss:
    if config.loss_name == "crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    if config.loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    raise ValueError(f"Unsupported loss: {config.loss_name}")


def build_mlp(
    config: ExperimentConfig,
    hidden_activation: str,
    output_activation: str = "softmax",
) -> tf.keras.Model:
    """Build the same two-hidden-layer MLP for all compared runs.

    Keeping layer width, random seed, loss, optimiser and epochs fixed lets us
    attribute most performance differences to the activation choice.  The
    default, Softmax at the output, is the proper probabilistic design for a
    single-label ten-class task such as MNIST.
    """

    for name, activation in (("hidden", hidden_activation), ("output", output_activation)):
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported {name} activation: {activation}")
    if config.loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported loss: {config.loss_name}")
    if config.optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.input_features,), name="pixels"),
            tf.keras.layers.Dense(
                config.hidden_units,
                activation=hidden_activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config.seed + 1),
                name="hidden_1",
            ),
            tf.keras.layers.Dense(
                config.hidden_units,
                activation=hidden_activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config.seed + 2),
                name="hidden_2",
            ),
            tf.keras.layers.Dense(
                config.classes,
                activation=output_activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config.seed + 3),
                name="digit_probability",
            ),
        ],
        name=f"mnist_{hidden_activation}_to_{output_activation}",
    )
    model.compile(
        optimizer=_make_optimizer(config),
        loss=_make_loss(config),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model
