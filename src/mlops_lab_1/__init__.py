"""Entrypoint for the mlops-lab-1 project, wired to the food11 training pipeline."""

from food11.train import main as train_main


def main() -> None:
    train_main()
