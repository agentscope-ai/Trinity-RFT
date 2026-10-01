#!/bin/bash
# run.sh — one-shot entry for the RL tutorial (docs/RL_tutorial).
#
# What it does:
#   1) load optional local secrets/paths from ./.env (see .env.example)
#   2) fail-fast on required env: ALFWORLD_DATA, TRINITY_MODEL_PATH
#   3) install locked deps (including FlashAttention2) into a Python 3.12 .venv
#      (Trinity-RFT is pulled in as a PINNED git dependency; see pyproject.toml)
#   4) generate the ALFWorld taskset jsonl if missing (needs $ALFWORLD_DATA)
#   5) start a single-node ray head and launch multi-step GRPO training
#
# Usage:
#   cp .env.example .env   # fill ALFWORLD_DATA + TRINITY_MODEL_PATH
#   bash run.sh
#
# Hardware: 8x GPU single node (4 train FSDP2 + 4 vLLM rollout), per the baseline yaml.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# --- 1) optional local paths file (gitignored) ---
if [ -f "$HERE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HERE/.env"
  set +a
fi

# --- 2) required paths (fail-fast; never hardcoded in the repo) ---
: "${ALFWORLD_DATA:?set ALFWORLD_DATA (path to alfworld json_2.1.1 data, see .env.example)}"
: "${TRINITY_MODEL_PATH:?set TRINITY_MODEL_PATH (path to a local Qwen3-1.7B checkpoint)}"
export ALFWORLD_DATA TRINITY_MODEL_PATH
export TRINITY_CHECKPOINT_ROOT_DIR="${TRINITY_CHECKPOINT_ROOT_DIR:-$HERE/checkpoints}"

# This is the platform covered by the pinned CUDA 13 FlashAttention wheel.
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo "[run] Training requires Linux x86_64 with NVIDIA GPUs; the pinned FlashAttention wheel uses Python 3.12." >&2
  exit 1
fi

# An explicit address means reuse only that cluster; never discover the latest
# unrelated Ray instance. Trinity's preflight reads RAY_ADDRESS, while the YAML
# reads TRINITY_RAY_ADDRESS. They must select the same cluster.
if [ -n "${RAY_ADDRESS:-}" ] && [ -n "${TRINITY_RAY_ADDRESS:-}" ] && \
   [ "$RAY_ADDRESS" != "$TRINITY_RAY_ADDRESS" ]; then
  echo "[run] RAY_ADDRESS and TRINITY_RAY_ADDRESS disagree; set just one concrete host:port." >&2
  exit 1
fi
ADDRESS="${TRINITY_RAY_ADDRESS:-${RAY_ADDRESS:-}}"
if [ "$ADDRESS" = auto ] || [ "$ADDRESS" = local ]; then
  echo "[run] Use a concrete Ray host:port, not auto/local, to avoid another experiment's cluster." >&2
  exit 1
fi
RAY_PORT="${TRINITY_RAY_PORT:-16379}"
if ! [[ "$RAY_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || [ "$RAY_PORT" -lt 1027 ] || [ "$RAY_PORT" -gt 65535 ]; then
  echo "[run] TRINITY_RAY_PORT must be an integer between 1027 and 65535." >&2
  exit 1
fi

# --- 3) install deps into local .venv ---
echo "[run] uv sync --locked --python 3.12 (public PyPI lock) ..."
# A shell-wide mirror setting must not invalidate or rewrite this public lock.
# Existing artifact URLs stay fixed; this is different from the LoRA pip setup.
uv sync --locked --python 3.12 --default-index https://pypi.org/simple
VENV="$HERE/.venv/bin"
export PATH="$VENV:$PATH"
# Run the driver directly, and also prevent the Ray uv hook when this script
# was itself launched by a parent `uv run` command.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# --- 4) generate ALFWorld taskset if missing ---
TASKSET="$HERE/examples/grpo_alfworld/alfworld_data/train.jsonl"
if [ ! -s "$TASKSET" ]; then
  echo "[run] generating ALFWorld taskset from \$ALFWORLD_DATA ..."
  "$VENV/python" examples/grpo_alfworld/get_alfworld_data.py \
    --game_data_path "$ALFWORLD_DATA" \
    --local_dir "$HERE/examples/grpo_alfworld/alfworld_data"
else
  echo "[run] taskset already present: $TASKSET"
fi

# --- 5) start ray head + launch training ---
if [ -n "$ADDRESS" ]; then
  export RAY_ADDRESS="$ADDRESS" TRINITY_RAY_ADDRESS="$ADDRESS"
  echo "[run] checking requested Ray cluster: $ADDRESS ..."
  "$VENV/ray" status --address "$ADDRESS"
else
  ADDRESS="127.0.0.1:$RAY_PORT"
  export RAY_ADDRESS="$ADDRESS" TRINITY_RAY_ADDRESS="$ADDRESS"
  echo "[run] starting Ray at $ADDRESS (local ports $((RAY_PORT - 3))-$RAY_PORT) ..."
  # Fail visibly on port conflicts. Never stop services owned by other runs.
  # Reuse a previously started tutorial cluster by explicitly setting
  # RAY_ADDRESS=127.0.0.1:16379 on the next invocation.
  "$VENV/ray" start --head --node-ip-address=127.0.0.1 --port="$RAY_PORT" \
    --ray-client-server-port="$((RAY_PORT - 1))" \
    --dashboard-agent-listen-port="$((RAY_PORT - 2))" \
    --dashboard-port="$((RAY_PORT - 3))" --include-dashboard=false \
    --min-worker-port=0 --max-worker-port=0
  "$VENV/ray" status --address "$ADDRESS"
fi

CONFIG="${CONFIG:-examples/grpo_alfworld_general_multi_step/alfworld.yaml}"
echo "[run] trinity run --config $CONFIG ..."
exec "$VENV/trinity" run --config "$CONFIG"
