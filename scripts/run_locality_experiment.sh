#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${repository_root}/scripts/runner/run_optimization.py"
config_root="${repository_root}/configs/optimization/experiments/locality"

usage() {
  echo "Usage: $0 {3|5|6|7|all} [runner arguments]"
  echo "Example: $0 5 --dry-run"
  echo "Example: $0 all --execution srun --max-parallel-runs 4"
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

experiment="$1"
shift

case "${experiment}" in
  3) configs=("03_random_three_positions.json") ;;
  5) configs=("05_tandem_top_p.json") ;;
  6) configs=("06_tandem_temperature.json") ;;
  7) configs=("07_random_mutang_retention.json") ;;
  all)
    configs=(
      "03_random_three_positions.json"
      "05_tandem_top_p.json"
      "06_tandem_temperature.json"
      "07_random_mutang_retention.json"
    )
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown locality experiment: ${experiment}" >&2
    usage >&2
    exit 2
    ;;
esac

cd "${repository_root}"
for config in "${configs[@]}"; do
  echo "Running locality experiment: ${config}"
  uv run python "${runner}" --config "${config_root}/${config}" "$@"
done
