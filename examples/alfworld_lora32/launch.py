#!/usr/bin/env python3
"""Prepare and run the single-node LoRA32 example. Uses stdlib until venv setup."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TRINITY = "65139711219da4338c954e546c4b8434e4f75ec2"
TUFT = "d4591c1bf707d14c3aaf238c63e35a1f3e88e442"
MODEL = "Qwen/Qwen3-1.7B"
CONFIG_DIR = ROOT / "examples/grpo_alfworld_general_multi_step"


class SetupError(Exception):
    pass


def chapter1_asset_environment(env):
    """Read only Chapter 1's two asset paths, without inheriting its run settings."""
    result = dict(env)
    names = ("TRINITY_MODEL_PATH", "ALFWORLD_DATA")
    path = ROOT / ".env"
    if path.is_file() and any(not result.get(name) for name in names):
        # Match run.sh's trusted shell .env syntax, but keep all other values
        # inside the child process. Capture diagnostics so keys cannot leak.
        command = 'set -e; source "$1" >/dev/null; printf "%s\\0%s\\0" "${TRINITY_MODEL_PATH:-}" "${ALFWORLD_DATA:-}"'
        child_env = {key: value for key, value in result.items() if key not in names}
        loaded = subprocess.run(["bash", "-c", command, "chapter1-paths", str(path)],
                                cwd=ROOT, env=child_env, capture_output=True, text=True)
        values = loaded.stdout.split("\0")
        if loaded.returncode or len(values) != 3 or values[-1] != "":
            raise SetupError("Cannot read the repository .env; check its shell syntax.")
        for name, value in zip(names, values):
            if not result.get(name) and value:
                result[name] = value
    return result


def repository_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def private_json(path, value):
    """Replace only helper-owned metadata; never print credentials."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SetupError(f"Cannot read metadata {path}; inspect it before continuing.") from exc


def process_identity(pid):
    """PID + boot + start time prevents reuse of stale PID records."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return [Path("/proc/sys/kernel/random/boot_id").read_text().strip(), stat[19]]
    except (OSError, IndexError):
        return None


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class Commands:
    def run(self, args, *, env=None, cwd=None, capture=False):
        try:
            return subprocess.run(
                [str(arg) for arg in args], check=True, env=env, cwd=cwd,
                text=True, timeout=120 if capture else None, stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
            ).stdout
        except subprocess.TimeoutExpired:
            raise SetupError(f"{Path(str(args[0])).name} timed out during a readiness or metadata check.") from None
        except subprocess.CalledProcessError as exc:
            # The command/exception may contain private package URLs or credentials.
            raise SetupError(f"{Path(str(args[0])).name} failed (exit {exc.returncode}). Check the preceding output or service log.") from None

    def spawn(self, args, *, env, cwd, log):
        with log.open("ab") as stream:
            os.chmod(log, 0o600)
            return subprocess.Popen(
                [str(arg) for arg in args], env=env, cwd=cwd,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )


class Example:
    def __init__(self, args, env=None, commands=None):
        self.args = args
        self.env = dict(os.environ if env is None else env)
        self.commands = commands or Commands()
        self.work = repository_path(self.env.get("LORA32_WORK_DIR") or "~/.cache/agentic-rl/alfworld-lora32")
        # Prevent a generated API key or checkpoint from accidentally entering this repo.
        if self.work == ROOT or ROOT in self.work.parents:
            raise SetupError("LORA32_WORK_DIR must be outside the tutorial checkout.")
        self.run_id = self.env.get("TRINITY_MACHINE_ID") or "reader72_001"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_id):
            raise SetupError("TRINITY_MACHINE_ID must use letters, digits, underscores or hyphens.")
        try:
            self.base = int(self.env.get("LORA32_PORT_BASE") or "16380")
        except ValueError:
            raise SetupError("LORA32_PORT_BASE must be an integer.") from None
        if not 1024 <= self.base <= 65514:
            raise SetupError("LORA32_PORT_BASE must be between 1024 and 65514.")
        self.client = self.work / "client-venv"
        self.source = self.work / "TuFT"
        self.server = self.source / ".venv"
        self.data = repository_path(self.env.get("ALFWORLD_DATA") or "alfworld_data")
        self.model = repository_path(self.env.get("TRINITY_MODEL_PATH") or "models/Qwen3-1.7B")
        self.output = self.work / "checkpoints/ALFWORLD" / ("Step_Wise_Alfworld_TuFT_lora32_lr5e5_speed_" + self.run_id)
        self.logs = self.work / "logs"
        # Ray uses Unix sockets: long NAS/home paths exceed their ~107-byte limit.
        self.ray_root = Path("/tmp") / f"alora32-{os.getuid()}-{digest(str(self.work))[:8]}"
        self.key = None

    def prerequisite_check(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise SetupError("This example targets Linux x86_64 with 8 x NVIDIA A100 80GB.")
        missing = [name for name in ("uv", "git", "redis-server", "wget", "nvidia-smi") if not shutil.which(name)]
        if missing:
            raise SetupError("Install these system prerequisites first: " + ", ".join(missing))
        rows = self.commands.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture=True,
        ).strip().splitlines()
        if len(rows) != 8 or any("A100" not in row or int(row.rsplit(",", 1)[1].strip()) < 79000 for row in rows):
            raise SetupError("The supplied configuration requires exactly 8 x A100 80GB; other hardware is not covered by this example.")
        print("[check] Linux x86_64, 8 x A100 80GB and system commands found.", flush=True)

    def check_output(self):
        if not self.args.resume:
            if self.output.exists():
                raise SetupError("Experiment output already exists. Use --resume, or choose a new TRINITY_MACHINE_ID.")
            return
        try:
            step = int((self.output / "latest_checkpointed_iteration.txt").read_text())
            sampler = int((self.output / "latest_state_dict_iteration.txt").read_text())
            required = [self.output / "trainer_meta.json",
                        self.output / f"global_step_{step}/.full_checkpoint",
                        self.output / f"global_step_{step}/remote_checkpoint_path.txt",
                        self.output / f"global_step_{sampler}/remote_sampler_path.txt"]
            if step < 0 or not all(p.is_file() for p in required):
                raise ValueError()
        except (OSError, ValueError):
            raise SetupError("Cannot resume: complete checkpoint and sampler metadata are required in this experiment's output.") from None
        if self.args.steps <= step:
            raise SetupError(f"--steps must exceed the saved checkpoint step ({step}); it is the total target.")

    def check_ports(self):
        owned = set()
        for path in (self.work / "services").glob("*.json"):
            state = read_json(path)
            if process_identity(state["pid"]) == state["identity"]:
                owned.update(state["ports"])
        for port in range(self.base, self.base + 21):
            if port_open(port) and port not in owned:
                raise SetupError("A configured port belongs to an unverified process. Choose another LORA32_PORT_BASE and work directory.")

    def source_check(self):
        if not self.source.exists():
            return False
        if not (self.source / ".git").exists():
            raise SetupError("The managed TuFT directory is incomplete or not a Git checkout; inspect it before continuing.")
        commit = self.commands.run(["git", "-C", self.source, "rev-parse", "HEAD"], capture=True).strip()
        dirty = self.commands.run(["git", "-C", self.source, "status", "--porcelain", "--untracked-files=normal"], env=dict(self.env, GIT_OPTIONAL_LOCKS="0"), capture=True).strip()
        if commit != TUFT or dirty:
            raise SetupError("The managed TuFT checkout changed. Restore it yourself or choose a new LORA32_WORK_DIR; no files were reset.")
        return True

    def environment_signature(self, directory):
        return digest(self.commands.run(["uv", "pip", "freeze", "--python", directory / "bin/python"], capture=True).encode())

    def any_managed_process_alive(self):
        for path in (self.work / "services").glob("*.json"):
            state = read_json(path)
            if process_identity(state["pid"]) == state["identity"]:
                return True
        return False

    def prepare_environments(self):
        spec = {"trinity": TRINITY, "tuft": TUFT, "requirements": digest((ROOT / "requirements-lora32.txt").read_bytes()), "installer": 1}
        marker = self.work / "environments.json"
        if marker.exists():
            recorded = read_json(marker)
            if recorded["spec"] != spec or not self.source_check():
                raise SetupError("Installed environment settings differ; use a new LORA32_WORK_DIR.")
            for name, path in (("client", self.client), ("server", self.server)):
                if not (path / "bin/python").is_file() or self.environment_signature(path) != recorded[name]:
                    raise SetupError(f"The {name} environment changed; use a new work directory rather than modifying a live service.")
            print("[setup] Reusing the fixed client and server environments.", flush=True)
            return recorded
        if self.any_managed_process_alive():
            raise SetupError("Services are running but installation metadata is missing; inspect the work directory first.")
        print("[setup] Installing the fixed Trinity and TuFT environments (first run may take a while).", flush=True)
        if not self.source.exists():
            self.commands.run(["git", "clone", "--branch", "main", "https://github.com/agentscope-ai/TuFT.git", self.source])
            self.commands.run(["git", "-C", self.source, "checkout", "--detach", TUFT])
        self.source_check()
        for path in (self.client, self.server):
            if not (path / "bin/python").exists():
                self.commands.run(["uv", "venv", "--python", "3.12", path])
        self.commands.run(["uv", "pip", "install", "--python", self.client / "bin/python", "--overrides", ROOT / "requirements-lora32.txt", "-r", ROOT / "requirements-lora32.txt"])
        self.commands.run(["uv", "pip", "install", "--python", self.server / "bin/python", "-e", str(self.source) + "[backend,persistence]", "tinker==0.25.0", "ray[default]==2.58.0", "transformers==5.16.1", "peft==0.20.0"])
        self.commands.run([self.server / "bin/python", self.source / "scripts/install_flash_attn.py", "--uv"], cwd=self.source)
        recorded = {"spec": spec, "client": self.environment_signature(self.client), "server": self.environment_signature(self.server)}
        private_json(marker, recorded)
        return recorded

    def data_ready(self):
        return (len(list((self.data / "json_2.1.1/train").glob("*/*/game.tw-pddl"))) == 3553
                and len(list((self.data / "json_2.1.1/valid_seen").glob("*/*/game.tw-pddl"))) == 140
                and (self.data / "logic/alfred.pddl").is_file()
                and (self.data / "logic/alfred.twl2").is_file())

    def model_ready(self):
        if not (self.model / "config.json").is_file() or not (self.model / "tokenizer_config.json").is_file():
            return False
        config = read_json(self.model / "config.json")
        expected = {"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28, "vocab_size": 151936}
        if any(config.get(key) != value for key, value in expected.items()):
            raise SetupError("TRINITY_MODEL_PATH does not match the Qwen3-1.7B model configuration; no files were overwritten.")
        index = self.model / "model.safetensors.index.json"
        if index.is_file():
            return all((self.model / name).is_file() for name in read_json(index)["weight_map"].values())
        return (self.model / "model.safetensors").is_file()

    def prepare_assets(self):
        env = dict(self.env, ALFWORLD_DATA=str(self.data))
        if not self.data_ready():
            print("[data] Downloading ALFWorld data; existing complete files are reused.", flush=True)
            self.commands.run([self.client / "bin/alfworld-download", "--data-dir", self.data], env=env)
        if not self.data_ready():
            raise SetupError("ALFWorld data must contain 3,553 train games, 140 valid_seen games, and logic files.")
        if not self.model_ready():
            print("[model] Downloading Qwen3-1.7B.", flush=True)
            self.commands.run([self.client / "bin/hf", "download", MODEL, "--local-dir", self.model], env=self.env)
        if not self.model_ready():
            raise SetupError("The Qwen3-1.7B download is incomplete.")
        # Dedicated tasksets avoid overwriting Chapter 1's absolute game paths.
        # Once generated, keep their ordering unchanged for checkpoint restoration.
        taskset = self.work / "taskset"
        complete = all((taskset / (split + ".jsonl")).is_file() for split in ("train", "test"))
        if not complete:
            if self.args.resume:
                raise SetupError("The saved taskset is missing; restore it before resuming.")
            self.commands.run([self.client / "bin/python", ROOT / "examples/grpo_alfworld/get_alfworld_data.py", "--game_data_path", self.data, "--local_dir", taskset], env=env)
        for split, count in (("train", 3553), ("test", 140)):
            rows = (taskset / (split + ".jsonl")).read_text().splitlines()
            if len(rows) != count or any(not Path(json.loads(row)["game_file"]).is_file()
                                         or self.data not in Path(json.loads(row)["game_file"]).parents for row in rows):
                raise SetupError("The saved taskset does not match ALFWORLD_DATA; use its original data directory or a new work directory.")

    def load_key(self):
        path = self.work / "api-key"
        if not path.exists():
            if (self.work / "services").exists() and list((self.work / "services").glob("*.json")):
                raise SetupError("The saved API key is missing; do not replace it while service state exists.")
            with path.open("x") as stream:
                os.chmod(path, 0o600)
                stream.write("tml-" + secrets.token_urlsafe(32) + "\n")
        if path.stat().st_mode & 0o077:
            raise SetupError("Set the saved api-key file permissions to 600 before continuing.")
        self.key = path.read_text().strip()
        if not self.key:
            raise SetupError("The saved API key is empty.")
        # Match Tinker 0.25's API-key/JWT prefix check without rotating saved credentials.
        if not self.key.startswith(("tml-", "eyJ")):
            raise SetupError(
                "The saved api-key file is incompatible with Tinker 0.25.0 "
                "(expected a 'tml-' key or JWT). It was not changed; use a new "
                "LORA32_WORK_DIR, or restore compatible credentials and their matching service state."
            )

    def environments(self):
        # Keep unrelated chapter/service variables out of both Ray heads.
        base = {k: v for k, v in self.env.items() if not k.startswith(("RAY_", "TUFT_", "TINKER_", "TRINITY_")) and k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES")}
        base.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", RAY_USAGE_STATS_ENABLED="0")
        client = dict(base, VIRTUAL_ENV=str(self.client), PATH=f"{self.client}/bin:{base.get('PATH', '')}",
                      ALFWORLD_DATA=str(self.data), TINKER_BASE_URL=f"http://127.0.0.1:{self.base + 1}",
                      TINKER_API_KEY=self.key, TINKER_TELEMETRY="0",
                      RAY_ADDRESS=f"127.0.0.1:{self.base + 12}", RAY_TMPDIR=str(self.ray_root / "client"))
        server = dict(base, VIRTUAL_ENV=str(self.server), PATH=f"{self.server}/bin:{base.get('PATH', '')}",
                      TUFT_HOME=str(self.work / "tuft-home"),
                      RAY_ADDRESS=f"127.0.0.1:{self.base + 2}", RAY_TMPDIR=str(self.ray_root / "server"))
        return client, server

    def write_config(self):
        path = self.work / "tuft-server.yaml"
        # Use the installed server's YAML parser; credentials travel on stdin-free
        # environment, never the command line. Capture output to keep the key private.
        code = '''import os, pathlib, yaml
p = pathlib.Path(os.environ["EXAMPLE_TEMPLATE"])
c = yaml.safe_load(p.read_text())
c["checkpoint_dir"] = os.environ["EXAMPLE_CHECKPOINTS"]
c["worker_venv_path"] = os.environ["EXAMPLE_SERVER_VENV"]
c["supported_models"][0]["model_path"] = os.environ["EXAMPLE_MODEL"]
c["authorized_users"] = {os.environ["EXAMPLE_API_KEY"]: "local-user"}
c["persistence"]["redis_url"] = os.environ["EXAMPLE_REDIS_URL"]
print(yaml.safe_dump(c, sort_keys=False))
'''
        env = dict(self.env, EXAMPLE_TEMPLATE=str(CONFIG_DIR / "tuft_lora32_server.example.yaml"),
                   EXAMPLE_CHECKPOINTS=str(self.work / "server-checkpoints"), EXAMPLE_SERVER_VENV=str(self.server),
                   EXAMPLE_MODEL=str(self.model), EXAMPLE_API_KEY=self.key, EXAMPLE_REDIS_URL=f"redis://127.0.0.1:{self.base}/0")
        content = self.commands.run([self.server / "bin/python", "-c", code], env=env, capture=True)
        if path.exists() and path.read_text() != content:
            raise SetupError("The stored TuFT configuration differs. Keep its original settings for resume, or choose a new work directory.")
        if not path.exists():
            with path.open("x") as stream:
                os.chmod(path, 0o600)
                stream.write(content)
        return path, digest(content.encode())

    def ray_command(self, directory, port, tempdir, gpus):
        # Ray's default worker range (10002-19999) includes our service ports.
        # Let the OS allocate worker ports instead of reserving that entire range.
        return [directory / "bin/ray", "start", "--head", "--block", f"--port={port}",
                "--node-ip-address=127.0.0.1", f"--num-gpus={gpus}", "--include-dashboard=false", "--disable-usage-stats",
                "--min-worker-port=0", "--max-worker-port=0",
                f"--temp-dir={tempdir}", f"--dashboard-agent-listen-port={port + 1}",
                f"--dashboard-agent-grpc-port={port + 2}", f"--runtime-env-agent-port={port + 3}",
                f"--metrics-export-port={port + 4}", f"--node-manager-port={port + 5}", f"--object-manager-port={port + 6}",
                f"--ray-client-server-port={port + 7}", f"--dashboard-port={port + 8}"]

    def service_state(self, name, fingerprint, ports):
        path = self.work / "services" / f"{name}.json"
        if path.exists():
            state = read_json(path)
            if state["fingerprint"] != fingerprint:
                raise SetupError(f"{name} settings or environment changed; refusing to reuse or restart it automatically.")
            if process_identity(state["pid"]) == state["identity"]:
                return state
        if any(port_open(port) for port in ports):
            raise SetupError(f"A {name} port is occupied by an unverified process. Choose another LORA32_PORT_BASE and work directory.")
        return None

    def ensure_service(self, name, command, env, ports, fingerprint, ready, timeout=180):
        state = self.service_state(name, fingerprint, ports)
        log = self.logs / f"{name}.log"
        if state is None:
            print(f"[service] Starting {name}; log: {log}", flush=True)
            process = self.commands.spawn(command, env=env, cwd=self.work, log=log)
            identity = process_identity(process.pid)
            if identity is None:
                raise SetupError(f"{name} exited during startup; inspect {log}.")
            state = {"pid": process.pid, "identity": identity, "fingerprint": fingerprint, "ports": ports}
            private_json(self.work / "services" / f"{name}.json", state)
        else:
            print(f"[service] Reusing {name} after configuration and process checks.", flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process_identity(state["pid"]) != state["identity"]:
                raise SetupError(f"{name} stopped before becoming ready; inspect {log}.")
            if ready():
                return
            time.sleep(2)
        raise SetupError(f"{name} did not become ready within {timeout}s; inspect {log}. It has not been killed.")

    def redis_ready(self):
        try:
            with socket.create_connection(("127.0.0.1", self.base), timeout=2) as sock:
                sock.sendall(b"*1\r\n$4\r\nPING\r\n")
                return sock.recv(64) == b"+PONG\r\n"
        except OSError:
            return False

    def ray_ready(self, directory, env):
        try:
            self.commands.run([directory / "bin/ray", "status", "--address", env["RAY_ADDRESS"]], env=env, capture=True)
            return True
        except SetupError:
            return False

    def tuft_ready(self):
        try:
            root = f"http://127.0.0.1:{self.base + 1}"
            for endpoint in ("healthz", "get_server_capabilities"):
                request = urllib.request.Request(root + "/api/v1/" + endpoint, headers={"X-API-Key": self.key})
                with urllib.request.urlopen(request, timeout=3) as response:
                    data = json.load(response)
                if endpoint == "healthz" and data.get("status") != "ok":
                    return False
                if endpoint == "get_server_capabilities" and not any(model.get("model_name") == MODEL for model in data.get("supported_models", [])):
                    return False
            return True
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def start_services(self, installed):
        client, server = self.environments()
        config, config_hash = self.write_config()
        # Hash service and accelerator settings; do not store their secret values.
        # Transient caller settings (PWD/SHLVL/terminal) are intentionally excluded.
        def fingerprint(kind, env, version):
            tracked = {k: v for k, v in env.items() if k.startswith(("TINKER_", "TUFT_", "RAY_", "NCCL_", "CUDA_", "TORCH_", "VLLM_", "OMP_")) or k in ("VIRTUAL_ENV", "ALFWORLD_DATA", "LD_LIBRARY_PATH", "PYTHONNOUSERSITE")}
            return digest([kind, version, tracked, config_hash])
        redis_fp = digest(["redis", self.base, str(self.work / "redis")])
        server_fp = fingerprint("ray-server", server, installed["server"])
        client_fp = fingerprint("ray-client", client, installed["client"])
        tuft_fp = fingerprint("tuft", server, installed["server"])
        # Validate all owned endpoints before starting anything new.
        for name, fp, ports in (("redis", redis_fp, [self.base]), ("ray-server", server_fp, list(range(self.base + 2, self.base + 11))),
                                ("ray-client", client_fp, list(range(self.base + 12, self.base + 21))), ("tuft", tuft_fp, [self.base + 1])):
            self.service_state(name, fp, ports)
        if self.service_state("tuft", tuft_fp, [self.base + 1]) is None:
            active = self.commands.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture=True).strip()
            if active:
                raise SetupError("GPU compute processes are already running. This example will not start another GPU service or stop them.")
        self.ensure_service("redis", ["redis-server", "--bind", "127.0.0.1", "--port", str(self.base), "--dir", self.work / "redis", "--appendonly", "yes"], server, [self.base], redis_fp, self.redis_ready)
        # Both heads inherit their own environment BEFORE any worker is created.
        self.ensure_service("ray-server", self.ray_command(self.server, self.base + 2, self.ray_root / "server", 8), server,
                            list(range(self.base + 2, self.base + 11)), server_fp, lambda: self.ray_ready(self.server, server))
        self.ensure_service("tuft", [self.server / "bin/tuft", "launch", "--host", "127.0.0.1", "--port", str(self.base + 1), "--config", config], server,
                            [self.base + 1], tuft_fp, self.tuft_ready, timeout=1800)
        self.ensure_service("ray-client", self.ray_command(self.client, self.base + 12, self.ray_root / "client", 0), client,
                            list(range(self.base + 12, self.base + 21)), client_fp, lambda: self.ray_ready(self.client, client))
        return client

    def check(self):
        self.prerequisite_check()
        self.check_output()
        self.check_ports()
        self.source_check()
        print(f"[check] Work directory: {self.work}")
        print(f"[check] Model: {self.model} ({'ready' if self.model_ready() else 'download required'})")
        print(f"[check] ALFWorld: {self.data} ({'ready' if self.data_ready() else 'download required'})")
        print(f"[check] Planned training: 72 runners, target {self.args.steps} steps, {'resume' if self.args.resume else 'fresh'}.")
        print("[check] Read-only preflight complete. No packages, services or training were started; GPU execution is not verified.")

    def run(self):
        self.prerequisite_check()
        self.check_output()
        self.check_ports()
        self.work.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.work, 0o700)
        with (self.work / ".setup.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SetupError("Another invocation is preparing or running this example in the same work directory.") from None
            for directory in (self.logs, self.work / "services", self.work / "redis", self.ray_root):
                directory.mkdir(exist_ok=True, mode=0o700)
                if directory.stat().st_uid != os.getuid():
                    raise SetupError("A managed directory is owned by another user; choose a different work directory.")
                os.chmod(directory, 0o700)
            installed = self.prepare_environments()
            self.prepare_assets()
            self.load_key()
            client = self.start_services(installed)
            client.update(TRINITY_MACHINE_ID=self.run_id, LORA32_VENV=str(self.client),
                          LORA32_MODEL_NAME=MODEL, LORA32_TASKSET_DIR=str(self.work / "taskset"),
                          TRINITY_CHECKPOINT_ROOT_DIR=str(self.work / "checkpoints"),
                          TRINITY_RAY_ADDRESS=client["RAY_ADDRESS"])
            command = ["bash", ROOT / "run_lora32_speed.sh", "--runners", "72", "--steps", str(self.args.steps)]
            if self.args.resume:
                command.append("--resume")
            print(f"[train] Output: {self.output}", flush=True)
            print(f"[train] Service logs: {self.logs}. Services and persistent state are retained after training.", flush=True)
            self.commands.run(command, env=client, cwd=ROOT)


def positive(value):
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare fixed LoRA32 environments, data and dedicated services; train with 72 runners.")
    parser.add_argument("--steps", type=positive, default=150, help="total training target (default: 150)")
    parser.add_argument("--resume", action="store_true", help="resume this experiment's complete checkpoint")
    parser.add_argument("--check", action="store_true", help="read-only prerequisite/path check; no installs, downloads, services or training")
    args = parser.parse_args(argv)
    try:
        example = Example(args, env=chapter1_asset_environment(os.environ))
        if args.check:
            example.check()
        else:
            example.run()
    except SetupError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Dedicated services and checkpoint state are retained; no global cleanup was run.", file=sys.stderr)
        return 130
    except (OSError, KeyError, ValueError):
        # Do not leak config values or subprocess environment via a traceback.
        print("[error] An OS operation or saved metadata is invalid. Inspect the work directory and service logs; no global cleanup was run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
