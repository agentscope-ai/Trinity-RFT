#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV}/bin/python"

cd "${REPO_ROOT}"

[[ -x "${PYTHON_BIN}" ]] || uv venv "${VENV}"

uv pip install \
    --python "${PYTHON_BIN}" \
    --overrides "${SCRIPT_DIR}/trinity-overrides.txt" \
    -e "${REPO_ROOT}[vllm,dev]"

"${PYTHON_BIN}" "${SCRIPT_DIR}/install_flash_attn.py" --uv

echo "Activate with: source ${VENV}/bin/activate"
