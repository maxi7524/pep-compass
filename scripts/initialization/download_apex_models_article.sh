#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
APEXGO_REPOSITORY="${APEXGO_REPOSITORY:-https://github.com/Yimeng-Zeng/APEXGo.git}"
TARGET_DIR="${APEX_MODELS_DIR:-${REPOSITORY_ROOT}/src/pep_compass/models/apex/APEX_pathogen_models}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ -e "${TARGET_DIR}" ]]; then
    echo "APEX model directory already exists: ${TARGET_DIR}" >&2
    echo "Remove it explicitly before downloading a fresh copy." >&2
    exit 1
fi

git clone --depth 1 --filter=blob:none --sparse \
    "${APEXGO_REPOSITORY}" "${TEMP_DIR}/APEXGo"
git -C "${TEMP_DIR}/APEXGo" sparse-checkout set \
    optimization/apex_oracle/APEX_pathogen_models

SOURCE_DIR="${TEMP_DIR}/APEXGo/optimization/apex_oracle/APEX_pathogen_models"
if [[ ! -d "${SOURCE_DIR}" ]] || [[ -z "$(find "${SOURCE_DIR}" -type f -print -quit)" ]]; then
    echo "APEX model weights were not found in the downloaded repository." >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "${TARGET_DIR}")"
cp -R -- "${SOURCE_DIR}" "${TARGET_DIR}"

echo "APEX model weights installed in: ${TARGET_DIR}"
