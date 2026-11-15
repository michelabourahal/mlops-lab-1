"""Quality gate for the `champion` alias of the registered `food11` model.

    uv run python src/food11/promote.py --version 8

The candidate only takes `champion` if its test accuracy beats the current champion's by
more than two standard errors; otherwise it is tagged `challenger` and `champion` stays put.
The decision and both accuracies are written as tags on the candidate version either way.
"""

from __future__ import annotations

import argparse
import math
import os

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "food11"
METRIC = "test_accuracy"
TEST_SIZE_PARAM = "test_samples"  # logged by train.py: size of the split test_accuracy was measured on
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def standard_error(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def should_promote(acc_candidate: float, acc_champion: float, n: int) -> bool:
    return acc_candidate >= acc_champion + 2 * standard_error(acc_champion, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a food11 model version if it passes the quality gate")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--n", type=int, help=f"test split size (default: the champion run's '{TEST_SIZE_PARAM}' param)")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=TRACKING_URI)

    def accuracy(run_id: str) -> float:
        metrics = client.get_run(run_id).data.metrics
        if METRIC not in metrics:
            raise SystemExit(f"run {run_id} has no '{METRIC}' metric")
        return metrics[METRIC]

    candidate = client.get_model_version(MODEL_NAME, args.version)
    acc_candidate = accuracy(candidate.run_id)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        champion = None

    tags = {"gate_candidate_accuracy": f"{acc_candidate:.4f}"}

    if champion is None:
        # Nothing to compare against: the first registered version becomes the champion.
        promote = True
        print(f"candidate  v{candidate.version}  {METRIC}={acc_candidate:.4f}")
        print("champion   none yet")
    elif champion.version == candidate.version:
        raise SystemExit(f"version {candidate.version} is already the champion")
    else:
        acc_champion = accuracy(champion.run_id)
        n = args.n or int(client.get_run(champion.run_id).data.params.get(TEST_SIZE_PARAM, 0))
        if n <= 0:
            raise SystemExit(f"champion run has no '{TEST_SIZE_PARAM}' param: pass the test split size with --n")
        se = standard_error(acc_champion, n)
        threshold = acc_champion + 2 * se
        promote = should_promote(acc_candidate, acc_champion, n)

        print(f"candidate  v{candidate.version}  {METRIC}={acc_candidate:.4f}")
        print(f"champion   v{champion.version}  {METRIC}={acc_champion:.4f}")
        print(f"n={n}  se={se:.4f}  threshold = champion + 2*se = {threshold:.4f}")
        tags |= {
            "gate_champion_accuracy": f"{acc_champion:.4f}",
            "gate_champion_version": str(champion.version),
            "gate_threshold": f"{threshold:.4f}",
        }

    alias = "champion" if promote else "challenger"
    client.set_registered_model_alias(MODEL_NAME, alias, candidate.version)
    tags["gate_decision"] = "promoted" if promote else "rejected"
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)

    print(f"decision   {tags['gate_decision']}: v{candidate.version} is now '{alias}'")


if __name__ == "__main__":
    main()
