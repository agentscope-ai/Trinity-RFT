"""CPU-only regressions for the full-parameter tutorial entrypoint.

Run with: python -m unittest discover -s scripts/rl_tutorial -p test_run.py -v
External commands are isolated stubs: no package installation, Ray, or GPU work.
"""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
FLASH_SHA = "eea423825f3e12818b98b2078e2cb5ce6fe6b73d22612316d2a55fad4701938f"
STUB = r'''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
location = str(pathlib.Path(sys.argv[0]).parent)
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as stream:
    stream.write(json.dumps({"name": name, "location": location, "args": args,
        "cwd": os.getcwd(), "env": {key: os.environ.get(key) for key in (
            "RAY_ADDRESS", "TRINITY_RAY_ADDRESS", "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
            "ALFWORLD_DATA", "TRINITY_MODEL_PATH", "PATH")}}) + "\n")
if name == "uname":
    print(os.environ.get("TEST_PLATFORM", "Linux") if args == ["-s"] else "x86_64")
elif name == "ray" and ".venv" not in location:
    print("WRONG SYSTEM RAY", file=sys.stderr)
    sys.exit(99)
elif name == "uv" and args[0] != "sync":
    print("unexpected uv run: workers must use the prepared environment", file=sys.stderr)
    sys.exit(99)
failure = os.environ.get("FAIL_COMMAND")
if failure == name or failure == name + " " + (args[0] if args else ""):
    print("original failure from " + failure, file=sys.stderr)
    sys.exit(37)
'''


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tutorial entry ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copy2(ROOT / "run.sh", self.root / "run.sh")
        self.bin = self.root / "fake-bin"
        self.venv = self.root / ".venv/bin"
        for directory, commands in (
            (self.bin, ("uname", "uv", "ray")),
            (self.venv, ("python", "ray", "trinity")),
        ):
            directory.mkdir(parents=True)
            for name in commands:
                script = directory / name
                script.write_text(f"#!{sys.executable}\n{STUB}")
                script.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.root),
            "COMMAND_LOG": str(self.log),
            "ALFWORLD_DATA": "data with spaces",
            "TRINITY_MODEL_PATH": "model with spaces",
        }

    def run_entry(self, **env):
        result = subprocess.run(
            ["/bin/bash", str(self.root / "run.sh")],
            cwd=self.root.parent,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_fresh_start_uses_locked_venv_and_same_explicit_cluster(self):
        result, calls = self.run_entry(UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/")
        self.assertEqual(result.returncode, 0, result.stderr)
        operations = [c for c in calls if c["name"] != "uname"]
        self.assertEqual([c["name"] for c in operations], ["uv", "python", "ray", "ray", "trinity"])
        self.assertEqual(operations[0]["args"], ["sync", "--locked", "--python", "3.12", "--default-index", "https://pypi.org/simple"])
        for call in operations[1:]:
            self.assertEqual(call["location"], str(self.venv))
            self.assertEqual(call["env"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"], "0")
        for call in operations[2:]:
            self.assertEqual(call["env"]["RAY_ADDRESS"], "127.0.0.1:16379")
            self.assertEqual(call["env"]["TRINITY_RAY_ADDRESS"], "127.0.0.1:16379")
        self.assertIn("--port=16379", operations[2]["args"])
        self.assertIn("--min-worker-port=0", operations[2]["args"])
        self.assertIn("--max-worker-port=0", operations[2]["args"])
        self.assertEqual(operations[3]["args"], ["status", "--address", "127.0.0.1:16379"])
        self.assertEqual(operations[1]["args"][2], "data with spaces")

    def test_requested_cluster_is_checked_without_starting_or_stopping(self):
        for variable in ("RAY_ADDRESS", "TRINITY_RAY_ADDRESS"):
            with self.subTest(variable=variable):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_entry(**{variable: "10.0.0.2:17000"})
                self.assertEqual(result.returncode, 0, result.stderr)
                ray = [c for c in calls if c["name"] == "ray"]
                self.assertEqual(len(ray), 1)
                self.assertEqual(ray[0]["args"], ["status", "--address", "10.0.0.2:17000"])
                self.assertEqual(calls[-1]["env"]["TRINITY_RAY_ADDRESS"], "10.0.0.2:17000")

    def test_conflicting_addresses_fail_before_installation(self):
        result, calls = self.run_entry(RAY_ADDRESS="host:1", TRINITY_RAY_ADDRESS="host:2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagree", result.stderr)
        self.assertFalse(any(c["name"] == "uv" for c in calls))

    def test_auto_and_local_cannot_select_an_unrelated_cluster(self):
        for value in ("auto", "local"):
            with self.subTest(value=value):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_entry(RAY_ADDRESS=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c["name"] == "uv" for c in calls))

    def test_ray_start_failure_preserves_error_and_does_not_launch_training(self):
        result, calls = self.run_entry(FAIL_COMMAND="ray start")
        self.assertEqual(result.returncode, 37)
        self.assertIn("original failure from ray start", result.stderr)
        self.assertFalse(any(c["name"] == "trinity" for c in calls))
        self.assertFalse(any(c["name"] == "ray" and c["args"][0] == "stop" for c in calls))

    def test_incompatible_requested_cluster_does_not_start_a_replacement(self):
        result, calls = self.run_entry(RAY_ADDRESS="host:16379", FAIL_COMMAND="ray status")
        self.assertEqual(result.returncode, 37)
        self.assertIn("original failure from ray status", result.stderr)
        self.assertEqual([c["args"][0] for c in calls if c["name"] == "ray"], ["status"])
        self.assertFalse(any(c["name"] == "trinity" for c in calls))

    def test_dependency_failure_stops_before_ray_or_training(self):
        result, calls = self.run_entry(FAIL_COMMAND="uv")
        self.assertEqual(result.returncode, 37)
        self.assertFalse(any(c["name"] in ("python", "ray", "trinity") for c in calls))

    def test_data_generation_failure_stops_before_ray(self):
        result, calls = self.run_entry(FAIL_COMMAND="python")
        self.assertEqual(result.returncode, 37)
        self.assertFalse(any(c["name"] in ("ray", "trinity") for c in calls))

    def test_existing_taskset_is_reused(self):
        taskset = self.root / "examples/grpo_alfworld/alfworld_data/train.jsonl"
        taskset.parent.mkdir(parents=True)
        taskset.write_text("{}\n")
        result, calls = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(c["name"] == "python" for c in calls))

    def test_empty_taskset_is_regenerated(self):
        taskset = self.root / "examples/grpo_alfworld/alfworld_data/train.jsonl"
        taskset.parent.mkdir(parents=True)
        taskset.touch()
        result, calls = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(c["name"] == "python" for c in calls))

    def test_unsupported_platform_fails_without_installing_gpu_packages(self):
        result, calls = self.run_entry(TEST_PLATFORM="Darwin")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Linux x86_64", result.stderr)
        self.assertFalse(any(c["name"] == "uv" for c in calls))

    def test_invalid_ray_port_fails_before_install(self):
        for value in ("0", "65536", "01000", "16379; exit 0"):
            with self.subTest(value=value):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_entry(TRINITY_RAY_PORT=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c["name"] == "uv" for c in calls))

    def test_custom_port_and_config_reach_the_intended_commands(self):
        result, calls = self.run_entry(TRINITY_RAY_PORT="17379", CONFIG="config with spaces.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--port=17379", next(c for c in calls if c["name"] == "ray")["args"])
        self.assertEqual(calls[-1]["args"], ["run", "--config", "config with spaces.yaml"])
        self.assertEqual(calls[-1]["env"]["RAY_ADDRESS"], "127.0.0.1:17379")

    def test_environment_file_and_driver_failure_are_preserved(self):
        (self.root / ".env").write_text("TRINITY_MODEL_PATH='local model'\n")
        result, calls = self.run_entry(FAIL_COMMAND="trinity")
        self.assertEqual(result.returncode, 37)
        self.assertEqual(calls[-1]["env"]["TRINITY_MODEL_PATH"], "local model")
        self.assertEqual(Path(calls[-1]["cwd"]).resolve(), self.root.resolve())


class DatasetRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data with spaces"
        self.output = self.root / "taskset"
        spec = importlib.util.spec_from_file_location(
            "tutorial_dataset", ROOT / "examples/grpo_alfworld/get_alfworld_data.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        for split in ("train", "valid_seen"):
            game = self.data / "json_2.1.1" / split / "task" / "trial" / "game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.touch()

    def test_parent_root_generates_real_absolute_game_paths(self):
        self.module.create_dataset_files(str(self.data), str(self.output))
        for split in ("train", "test"):
            rows = [json.loads(line) for line in (self.output / f"{split}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(rows[0]["game_file"]).is_file())
            self.assertTrue(Path(rows[0]["game_file"]).is_absolute())

    def test_json_subdirectory_is_rejected_without_creating_empty_taskset(self):
        with self.assertRaisesRegex(ValueError, "dataset root"):
            self.module.create_dataset_files(str(self.data / "json_2.1.1"), str(self.output))
        self.assertFalse(self.output.exists())

    def test_missing_split_is_rejected_without_overwriting_existing_data(self):
        shutil.rmtree(self.data / "json_2.1.1/valid_seen")
        self.output.mkdir()
        saved = self.output / "train.jsonl"
        saved.write_text("preserved\n")
        with self.assertRaisesRegex(ValueError, "valid_seen"):
            self.module.create_dataset_files(str(self.data), str(self.output))
        self.assertEqual(saved.read_text(), "preserved\n")


class LockedConfigurationTests(unittest.TestCase):
    def test_baselines_use_tensorboard_and_direct_actions(self):
        for path in ("examples/grpo_alfworld_general_multi_step/alfworld.yaml", "examples/gigpo_alfworld/gigpo.yaml"):
            with self.subTest(path=path):
                config = yaml.safe_load((ROOT / path).read_text())
                self.assertEqual(config["monitor"]["monitor_type"], "tensorboard")
                self.assertIs(config["model"]["enable_thinking"], False)
                self.assertIn("TRINITY_RAY_ADDRESS", config["cluster"]["ray_address"])

    def test_dynamic_batch_accepts_the_long_sample_and_the_model_limit(self):
        # The pinned verl FSDP path requires max_token_len >= max_seq_len.
        for path in ("examples/grpo_alfworld_general_multi_step/alfworld.yaml", "examples/gigpo_alfworld/gigpo.yaml"):
            with self.subTest(path=path):
                config = yaml.safe_load((ROOT / path).read_text())
                budget = config["trainer"]["max_token_len_per_gpu"]
                self.assertGreaterEqual(budget, 16828)  # Real failing sample from run6.
                self.assertGreaterEqual(budget, config["model"]["max_model_len"])

    def test_flash_attention_wheel_is_locked_and_platform_scoped(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
        requirement = next(d for d in project["project"]["dependencies"] if d.startswith("flash-attn"))
        self.assertIn("sys_platform == 'linux'", requirement)
        self.assertIn("platform_machine == 'x86_64'", requirement)
        package = next(p for p in lock["package"] if p["name"] == "flash-attn")
        self.assertEqual(package["wheels"][0]["hash"], f"sha256:{FLASH_SHA}")
        self.assertIn("cp312-cp312-linux_x86_64.whl", package["wheels"][0]["url"])
        versions = {p["name"]: p["version"] for p in lock["package"]}
        for name, expected in {"torch": "2.11.0", "verl": "0.9.0", "vllm": "0.23.0", "transformers": "5.16.1", "ray": "2.58.0", "alfworld": "0.4.2"}.items():
            self.assertEqual(versions[name], expected)


if __name__ == "__main__":
    unittest.main()
