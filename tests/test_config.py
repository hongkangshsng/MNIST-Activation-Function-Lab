from src.config import ExperimentConfig, SUPPORTED_ACTIVATIONS


def test_mnist_shape_constants() -> None:
    config = ExperimentConfig()
    assert config.input_features == 784
    assert config.classes == 10
    assert {"sigmoid", "tanh", "relu", "softplus", "softmax"} <= set(SUPPORTED_ACTIVATIONS)
