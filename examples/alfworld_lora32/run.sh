#!/usr/bin/env bash
# One-command LoRA32 example; the helper manages only this example's resources.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
# --help does not source local shell configuration.
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then
    exec python3 "$HERE/launch.py" "$@"
  fi
done
# Explicit nonempty paths win over the optional LoRA file. Empty entries in
# that file must not erase inherited paths from the caller or Chapter 1.
explicit_model="${TRINITY_MODEL_PATH:-}"
explicit_data="${ALFWORLD_DATA:-}"
cd "$ROOT"
if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi
export TRINITY_MODEL_PATH="${explicit_model:-${TRINITY_MODEL_PATH:-}}"
export ALFWORLD_DATA="${explicit_data:-${ALFWORLD_DATA:-}}"
exec python3 "$HERE/launch.py" "$@"
