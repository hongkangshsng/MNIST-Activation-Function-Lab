import numpy as np
import pytest

from src.data import split_train_validation


def test_split_is_disjoint_reproducible_and_respects_limit() -> None:
    # The first column acts as a stable sample identifier for overlap checks.
    x = np.arange(100, dtype=np.float32).reshape(50, 2)
    y = np.arange(50, dtype=np.int64)

    first = split_train_validation(x, y, validation_fraction=0.2, seed=7, train_limit=30)
    second = split_train_validation(x, y, validation_fraction=0.2, seed=7, train_limit=30)
    x_fit, y_fit, x_val, y_val = first

    assert x_fit.shape == (24, 2)
    assert x_val.shape == (6, 2)
    assert len(y_fit) + len(y_val) == 30
    assert set(y_fit).isdisjoint(set(y_val))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_split_rejects_invalid_inputs() -> None:
    x = np.arange(10).reshape(5, 2)
    y = np.arange(4)
    with pytest.raises(ValueError, match="same number"):
        split_train_validation(x, y, validation_fraction=0.2, seed=1)
    with pytest.raises(ValueError, match="validation_fraction"):
        split_train_validation(x, np.arange(5), validation_fraction=1.0, seed=1)
