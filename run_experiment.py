"""Command-line entry point for the MNIST activation-function lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare activation functions on MNIST with an MLP.")
    parser.add_argument("--suite", choices=("hidden", "legacy", "both"), default="hidden")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--loss", choices=("crossentropy", "mse"), default="crossentropy")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--train-limit", type=int, default=None, help="Use fewer training samples for a quick check.")
    parser.add_argument("--test-limit", type=int, default=None, help="Use fewer test samples for a quick check.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the official training split held out for validation (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Delayed import lets `python run_experiment.py --help` work before the
    # optional virtual environment has installed TensorFlow.
    from src.experiment import run_suite

    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary = run_suite(config, args.suite)
    print("\n=== Final ranking ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts saved in: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
