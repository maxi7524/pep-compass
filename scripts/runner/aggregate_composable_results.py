"""Aggregate atomic results produced by composable experiment workers."""

import argparse
import json
from pathlib import Path

from pep_compass.experiments.aggregation import aggregate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate_experiment(args.output_directory), indent=2))


if __name__ == "__main__":
    main()
