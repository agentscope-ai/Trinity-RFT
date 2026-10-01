#!/usr/bin/env bash
# Launch the independent Trinity client. Does not install packages or manage services.
# Default: 72 runners, fresh output, up to 150 steps. See reproduction guide.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_lora32_speed.sh [--runners 16|32|48|72] [--steps N] [--resume]

Required environment:
  TRINITY_MACHINE_ID  A unique name for this experiment (keep it when resuming).
  ALFWORLD_DATA      Absolute path to downloaded ALFWorld data.
  TINKER_API_KEY     The key configured in your TuFT service.

Optional environment:
  LORA32_VENV                    Client environment (default: .venv-lora32).
  LORA32_TASKSET_DIR             Optional dedicated taskset directory.
  LORA32_MODEL_NAME              TuFT model name (default: Qwen/Qwen3-1.7B).
  TRINITY_CHECKPOINT_ROOT_DIR    Output root (default: checkpoints-lora32).
  TRINITY_RAY_ADDRESS            Existing client Ray head (default: 127.0.0.1:6395).
  TINKER_BASE_URL                Existing TuFT service (default: http://127.0.0.1:10610).

Defaults: 72 runners, 150 total steps. --steps 250 selects the historical horizon.
New runs refuse an existing output directory. --resume reuses the latest saved state;
its target --steps must exceed the checkpoint step.
The script never stops or restarts Ray/TuFT, and never changes the tutorial's .venv.
EOF
}

RUNNERS=72
STEPS=150
MODE=fresh
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runners)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RUNNERS="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      STEPS="$2"
      shift 2
      ;;
    --resume) MODE=resume; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$RUNNERS" in
  16|32|48|72) ;;
  *) printf 'Runners must be 16, 32, 48, or 72.\n' >&2; exit 2 ;;
esac
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Steps must be a positive integer.\n' >&2
  exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
: "${TRINITY_MACHINE_ID:?Set a unique experiment name, for example reader72_001}"
: "${ALFWORLD_DATA:?Set the absolute path to downloaded ALFWorld data}"
: "${TINKER_API_KEY:?Set the key configured in your TuFT service}"
if [[ ! "$TRINITY_MACHINE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  printf 'TRINITY_MACHINE_ID must contain only letters, digits, underscores, and hyphens.\n' >&2
  exit 2
fi

LORA32_VENV="${LORA32_VENV:-$HERE/.venv-lora32}"
PYTHON="$LORA32_VENV/bin/python"
[[ -x "$PYTHON" ]] || { printf 'Install the independent LoRA32 environment first: %s\n' "$LORA32_VENV" >&2; exit 2; }
export TRINITY_MACHINE_ID ALFWORLD_DATA TINKER_API_KEY
export TRINITY_CHECKPOINT_ROOT_DIR="${TRINITY_CHECKPOINT_ROOT_DIR:-$HERE/checkpoints-lora32}"
export TRINITY_MODEL_PATH="${LORA32_MODEL_NAME:-Qwen/Qwen3-1.7B}"
export TRINITY_RAY_ADDRESS="${TRINITY_RAY_ADDRESS:-127.0.0.1:6395}"
export RAY_ADDRESS="$TRINITY_RAY_ADDRESS"
export TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:10610}"
export TINKER_TELEMETRY=0
export PYTHONUNBUFFERED=1

CONFIG="$HERE/examples/grpo_alfworld_general_multi_step/alfworld_tuft_lora32_lr5e5_speed.yaml"
if [[ "$RUNNERS" != 16 ]]; then
  CONFIG="${CONFIG%.yaml}_ab_r${RUNNERS}.yaml"
fi

exec "$PYTHON" - "$CONFIG" "$MODE" "$RUNNERS" "$STEPS" <<'PY'
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import yaml
from omegaconf import OmegaConf

config, mode, runners, steps_arg = sys.argv[1:]
steps = int(steps_arg)
expected = "65139711219da4338c954e546c4b8434e4f75ec2"
try:
    distribution = importlib.metadata.distribution("trinity-rft")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
except (importlib.metadata.PackageNotFoundError, ValueError) as exc:
    raise SystemExit("Install requirements-lora32.txt into the independent client environment.") from exc
if origin.get("vcs_info", {}).get("commit_id") != expected:
    raise SystemExit("The client environment is not pinned to the documented upstream Trinity commit.")

data_root = Path(os.environ["ALFWORLD_DATA"])
if not data_root.is_absolute() or not (data_root / "json_2.1.1").is_dir():
    raise SystemExit("ALFWORLD_DATA must be an absolute path containing json_2.1.1/.")
taskset_dir = Path(os.environ.get("LORA32_TASKSET_DIR", "examples/grpo_alfworld/alfworld_data")).expanduser().resolve()
taskset = taskset_dir / "train.jsonl"
if not taskset.is_file():
    raise SystemExit("Generate the local ALFWorld taskset first; see the reproduction guide.")
with taskset.open() as stream:
    first = stream.readline()
if not first or not Path(json.loads(first)["game_file"]).is_file():
    raise SystemExit("The taskset is empty or contains paths from another machine; regenerate it.")

output_root = Path(os.environ["TRINITY_CHECKPOINT_ROOT_DIR"]).expanduser().resolve()
os.environ["TRINITY_CHECKPOINT_ROOT_DIR"] = str(output_root)
run_name = "Step_Wise_Alfworld_TuFT_lora32_lr5e5_speed_" + os.environ["TRINITY_MACHINE_ID"]
run_dir = output_root / "ALFWORLD" / run_name
if mode == "fresh":
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise SystemExit("Output already exists. Choose a new experiment name, or explicitly use --resume.")
else:
    try:
        step = int((run_dir / "latest_checkpointed_iteration.txt").read_text().strip())
        checkpoint = run_dir / f"global_step_{step}"
        required = [checkpoint / ".full_checkpoint", checkpoint / "remote_checkpoint_path.txt",
                    run_dir / "latest_state_dict_iteration.txt", run_dir / "trainer_meta.json"]
        if not all(p.is_file() for p in required):
            raise ValueError("incomplete checkpoint metadata")
        sampler_step = int((run_dir / "latest_state_dict_iteration.txt").read_text().strip())
        sampler = run_dir / f"global_step_{sampler_step}" / "remote_sampler_path.txt"
        if not sampler.is_file():
            raise ValueError("missing sampler pointer")
        if steps <= step:
            raise ValueError(f"target --steps {steps} must exceed saved checkpoint step {step}")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot resume: {exc}. Check the saved state and server checkpoint registry.") from exc
    print(f"Resuming saved training state {step}; sampler pointer is {sampler_step}.", flush=True)

# Keep a process lock across exec, so a second launcher cannot write this run.
lock_fd = os.open(run_dir / ".launcher.lock", os.O_CREAT | os.O_RDWR, 0o600)
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Another launcher already holds this experiment's output lock.")
os.set_inheritable(lock_fd, True)

# Preserve the checked-in templates. Record this invocation's effective settings,
# including the chosen teaching horizon and resolved environment paths.
with Path(config).open() as stream:
    effective = yaml.safe_load(stream)
effective["trainer"]["total_steps"] = steps
effective["buffer"]["explorer_input"]["taskset"]["path"] = str(taskset_dir)
effective = OmegaConf.to_container(OmegaConf.create(effective), resolve=True)
config_dir = run_dir / "launch_configs"
config_dir.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
effective_path = config_dir / f"{stamp}_{mode}_r{runners}_steps{steps}.yaml"
effective_path.write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True))
print(f"LoRA32: {runners} runners, {mode}, target {steps} steps; output: {run_dir}", flush=True)
print(f"Recorded config: {effective_path}", flush=True)
os.execv(sys.executable, [sys.executable, "-m", "trinity.cli.launcher", "run", "--config", str(effective_path)])
PY
