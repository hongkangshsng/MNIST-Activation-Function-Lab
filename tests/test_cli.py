import sys

from run_experiment import parse_args


def test_cli_accepts_validation_fraction(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--validation-fraction", "0.2"])
    assert parse_args().validation_fraction == 0.2
