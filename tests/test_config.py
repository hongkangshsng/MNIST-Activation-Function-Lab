import pytest

from src.config import ExperimentConfig, SUPPORTED_ACTIVATIONS


def test_mnist_shape_constants() -> None:
    config = ExperimentConfig()
    assert config.input_features == 784
    assert config.classes == 10
    assert {"sigmoid", "tanh", "relu", "softplus", "softmax"} <= set(SUPPORTED_ACTIVATIONS)


def test_validation_fraction_must_leave_a_holdout_split() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        ExperimentConfig(validation_fraction=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        ExperimentConfig(validation_fraction=1.0)
