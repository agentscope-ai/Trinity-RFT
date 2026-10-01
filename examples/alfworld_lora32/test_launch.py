"""Offline orchestration checks. No GPU installs, network, real services or training.

Run: python3 -m unittest discover -s examples/alfworld_lora32 -p 'test_*.py' -v
"""
import argparse
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch


class FakeCommands:
    def __init__(self, example):
        self.example = example
        self.calls = []
        self.spawns = []
        self.live = {}
        self.ports = set()
        self.dirty = False
        self.busy_gpu = False
        self.install_failure = False
        self.freeze = "fixed-package==1\n"

    def run(self, args, *, env=None, cwd=None, capture=False):
        args = list(map(str, args))
        self.calls.append((args, dict(env or {})))
        example = self.example
        command = Path(args[0]).name
        if command == "nvidia-smi":
            return "9999\n" if self.busy_gpu and "--query-compute-apps=pid" in args else ("" if "--query-compute-apps=pid" in args else "NVIDIA A100-SXM4-80GB, 81920\n" * 8)
        if command == "git":
            if "clone" in args:
                (example.source / ".git").mkdir(parents=True)
            if "rev-parse" in args:
                return launch.TUFT + "\n"
            if "status" in args:
                return " M src/tuft/server.py\n" if self.dirty else ""
            return ""
        if command == "uv":
            if "venv" in args:
                path = Path(args[-1]) / "bin/python"
                path.parent.mkdir(parents=True)
                path.touch()
            if "freeze" in args:
                return self.freeze
            if "install" in args and self.install_failure:
                self.install_failure = False
                raise launch.SetupError("Simulated package download failure")
            return ""
        if command == "alfworld-download":
            for split, size in (("train", 3553), ("valid_seen", 140)):
                for i in range(size):
                    game = example.data / "json_2.1.1" / split / str(i) / "game/game.tw-pddl"
                    game.parent.mkdir(parents=True)
                    game.touch()
            (example.data / "logic").mkdir()
            for filename in ("alfred.pddl", "alfred.twl2"):
                (example.data / "logic" / filename).touch()
            return ""
        if command == "hf":
            example.model.mkdir(parents=True)
            for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (example.model / filename).write_text(json.dumps({"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28, "vocab_size": 151936}))
            return ""
        if command == "python" and "-c" in args:
            # The production process uses PyYAML. This fixture emits valid JSON/YAML
            # without importing GPU packages or requiring PyYAML in the test runner.
            return json.dumps({"checkpoint_dir": env["EXAMPLE_CHECKPOINTS"], "worker_venv_path": env["EXAMPLE_SERVER_VENV"],
                               "model_path": env["EXAMPLE_MODEL"], "authorized_users": {env["EXAMPLE_API_KEY"]: "local-user"},
                               "redis_url": env["EXAMPLE_REDIS_URL"]})
        if command == "python" and args[1].endswith("get_alfworld_data.py"):
            spec = importlib.util.spec_from_file_location("data_generator", args[1])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.create_dataset_files(str(example.data), str(example.work / "taskset"))
        return ""

    def spawn(self, args, *, env, cwd, log):
        args = list(map(str, args))
        pid = 100 + len(self.spawns)
        self.spawns.append((args, dict(env)))
        self.live[pid] = ["fake-boot", str(pid)]
        for arg in args:
            if arg.startswith("--port="):
                self.ports.add(int(arg.split("=")[1]))
        if "--port" in args:
            self.ports.add(int(args[args.index("--port") + 1]))
        return argparse.Namespace(pid=pid)


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        args = argparse.Namespace(steps=150, resume=False, check=False)
        self.example = launch.Example(args, env={"LORA32_WORK_DIR": str(Path(self.temporary.name) / "work"),
                                               "ALFWORLD_DATA": str(Path(self.temporary.name) / "data"),
                                               "TRINITY_MODEL_PATH": str(Path(self.temporary.name) / "model"),
                                               "PATH": "/usr/bin"})
        # Keep even mocked Ray temp directories under the fixture.
        self.example.ray_root = Path(self.temporary.name) / "ray"
        self.fake = FakeCommands(self.example)
        self.example.commands = self.fake
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(launch.platform, "system", return_value="Linux"))
        self.stack.enter_context(patch.object(launch.platform, "machine", return_value="x86_64"))
        self.stack.enter_context(patch.object(launch.shutil, "which", side_effect=lambda name: "/mock/" + name))
        self.stack.enter_context(patch.object(launch, "process_identity", side_effect=lambda pid: self.fake.live.get(pid)))
        self.stack.enter_context(patch.object(launch, "port_open", side_effect=lambda port: port in self.fake.ports))
        self.stack.enter_context(patch.object(self.example, "redis_ready", return_value=True))
        self.stack.enter_context(patch.object(self.example, "tuft_ready", return_value=True))
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def checkpoint(self, step):
        root = self.example.output
        (root / f"global_step_{step}").mkdir(parents=True)
        for name in ("latest_checkpointed_iteration.txt", "latest_state_dict_iteration.txt"):
            (root / name).write_text(str(step))
        (root / "trainer_meta.json").write_text("{}")
        for name in (".full_checkpoint", "remote_checkpoint_path.txt", "remote_sampler_path.txt"):
            (root / f"global_step_{step}" / name).write_text("saved")

    def test_fresh_install_services_resume_and_reuse(self):
        self.example.run()
        spawned = self.fake.spawns
        self.assertEqual([Path(args[0]).name for args, _ in spawned], ["redis-server", "ray", "tuft", "ray"])
        self.assertIn("--num-gpus=8", spawned[1][0])
        self.assertIn("--num-gpus=0", spawned[3][0])
        client_env = spawned[3][1]
        self.assertNotIn("TINKER_REF_LOGPROB_CONCURRENCY", client_env)
        self.assertNotIn("TINKER_TIMEOUT", client_env)
        self.assertNotEqual(client_env["RAY_ADDRESS"], spawned[1][1]["RAY_ADDRESS"])
        self.assertNotIn("TUFT_FSDP_GPUS", spawned[1][1])
        training_command, training_env = self.fake.calls[-1]
        self.assertEqual(training_command[-4:], ["--runners", "72", "--steps", "150"])
        self.assertEqual(training_env["LORA32_TASKSET_DIR"], str(self.example.work / "taskset"))
        key = self.example.key
        self.assertTrue(key.startswith("tml-"))
        self.assertNotIn(key, self.output.getvalue())
        self.assertNotIn(key, json.dumps([args for args, _ in self.fake.calls + spawned]))
        for path in (self.example.work / "services").glob("*.json"):
            self.assertNotIn(key, path.read_text())
        self.assertEqual((self.example.work / "api-key").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.example.work / "tuft-server.yaml").stat().st_mode & 0o777, 0o600)
        # The training command is mocked; create a synthetic saved checkpoint.
        self.checkpoint(150)
        self.example.args.resume = True
        self.example.args.steps = 250
        previous_installs = len([args for args, _ in self.fake.calls if "install" in args])
        self.example.run()
        self.assertEqual(len(self.fake.spawns), 4)
        self.assertEqual(len([args for args, _ in self.fake.calls if "install" in args]), previous_installs)
        self.assertEqual(self.fake.calls[-1][0][-3:], ["--steps", "250", "--resume"])
        self.assertEqual(self.example.key, key)
        self.assertEqual(len([a for a, _ in self.fake.calls if any("get_alfworld_data.py" in p for p in a)]), 1)
        self.fake.freeze = "changed-package==2\n"
        with self.assertRaisesRegex(launch.SetupError, "environment changed"):
            self.example.prepare_environments()

    def test_incompatible_saved_key_is_not_rotated_or_disclosed(self):
        self.example.work.mkdir()
        path = self.example.work / "api-key"
        saved = b"legacy-private-key-without-prefix\n"
        path.write_bytes(saved)
        path.chmod(0o600)
        services = self.example.work / "services"
        services.mkdir()
        state = services / "tuft.json"
        state.write_text('{"retained": true}')
        for service_state_exists in (True, False):
            with self.subTest(service_state_exists=service_state_exists):
                if not service_state_exists:
                    state.unlink()
                with patch.object(launch.secrets, "token_urlsafe") as generate:
                    with self.assertRaisesRegex(launch.SetupError, "Tinker 0.25.0") as error:
                        self.example.load_key()
                    generate.assert_not_called()
                self.assertIn("It was not changed", str(error.exception))
                self.assertIn("LORA32_WORK_DIR", str(error.exception))
                self.assertNotIn(saved.decode().strip(), str(error.exception))
                self.assertEqual(path.read_bytes(), saved)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                if service_state_exists:
                    self.assertEqual(state.read_text(), '{"retained": true}')

    def test_compatible_saved_keys_are_reused(self):
        self.example.work.mkdir()
        path = self.example.work / "api-key"
        for key in ("tml-saved-private-key", "eyJfixture-jwt"):
            with self.subTest(prefix=key[:3]):
                saved = (key + "\n").encode()
                path.write_bytes(saved)
                path.chmod(0o600)
                with patch.object(launch.secrets, "token_urlsafe") as generate:
                    self.example.load_key()
                    generate.assert_not_called()
                self.assertEqual(self.example.key, key)
                self.assertEqual(path.read_bytes(), saved)

    @unittest.skipUnless(importlib.util.find_spec("tinker"), "Requires the pinned Tinker SDK; no network")
    def test_generated_key_passes_real_tinker_auth_provider(self):
        if importlib.metadata.version("tinker") != "0.25.0":
            self.skipTest("Requires the example's Tinker 0.25.0")
        from tinker.lib._auth_token_provider import ApiKeyAuthProvider

        self.example.work.mkdir()
        self.example.load_key()
        # Exercise only the SDK's local auth provider, never a service client.
        provider = ApiKeyAuthProvider(api_key=self.example.key)
        self.assertEqual(asyncio.run(provider.get_token()), self.example.key)

    def test_read_only_check_creates_nothing(self):
        self.example.check()
        self.assertFalse(self.example.work.exists())
        self.assertEqual(len(self.fake.calls), 1)  # nvidia-smi only
        self.assertFalse(self.fake.spawns)

    def test_existing_output_and_bad_resume_fail_before_install(self):
        self.checkpoint(150)
        with self.assertRaisesRegex(launch.SetupError, "already exists"):
            self.example.run()
        self.example.args.resume = True
        with self.assertRaisesRegex(launch.SetupError, "must exceed"):
            self.example.run()
        self.example.args.steps = 250
        (self.example.output / "global_step_150/remote_checkpoint_path.txt").unlink()
        with self.assertRaisesRegex(launch.SetupError, "complete checkpoint"):
            self.example.run()
        self.assertFalse(self.fake.spawns)
        self.assertFalse(any("install" in args for args, _ in self.fake.calls))

    def test_unknown_port_is_never_reused(self):
        self.fake.ports.add(self.example.base)
        with self.assertRaisesRegex(launch.SetupError, "unverified"):
            self.example.service_state("redis", "fingerprint", [self.example.base])
        self.assertFalse(self.fake.spawns)

    def test_fingerprint_change_refuses_running_service(self):
        directory = self.example.work / "services"
        directory.mkdir(parents=True)
        self.fake.live[123] = ["boot", "start"]
        launch.private_json(directory / "tuft.json", {"pid": 123, "identity": ["boot", "start"], "fingerprint": "old"})
        with self.assertRaisesRegex(launch.SetupError, "changed"):
            self.example.service_state("tuft", "new", [self.example.base + 1])
        self.assertFalse(self.fake.spawns)

    def test_recycled_pid_is_not_reused(self):
        directory = self.example.work / "services"
        directory.mkdir(parents=True)
        self.fake.live[123] = ["boot", "new-start"]
        launch.private_json(directory / "redis.json", {"pid": 123, "identity": ["boot", "old-start"], "fingerprint": "same"})
        self.fake.ports.add(self.example.base)
        with self.assertRaisesRegex(launch.SetupError, "unverified"):
            self.example.service_state("redis", "same", [self.example.base])

    def test_failed_preparation_retries_without_resetting_checkout(self):
        self.example.work.mkdir()
        self.fake.install_failure = True
        with self.assertRaisesRegex(launch.SetupError, "Simulated"):
            self.example.prepare_environments()
        self.assertFalse((self.example.work / "environments.json").exists())
        self.example.prepare_environments()
        self.assertEqual(len([a for a, _ in self.fake.calls if "clone" in a]), 1)
        self.assertFalse(any("reset" in a for a, _ in self.fake.calls))
        self.fake.dirty = True
        with self.assertRaisesRegex(launch.SetupError, "checkout changed"):
            self.example.prepare_environments()

    def test_all_ray_ports_disjoint(self):
        server = self.example.ray_command(self.example.server, self.example.base + 2, "tmp-server", 8)
        client = self.example.ray_command(self.example.client, self.example.base + 12, "tmp-client", 0)
        ports = lambda args: {int(str(arg).split("=")[1]) for arg in args if "port=" in str(arg) and int(str(arg).split("=")[1]) != 0}
        self.assertEqual(len(ports(server)), 9)
        self.assertEqual(len(ports(client)), 9)
        self.assertFalse(ports(server) & ports(client))
        self.assertNotIn(self.example.base + 1, ports(client) | ports(server))

    @unittest.skipUnless(importlib.util.find_spec("ray"), "Requires the pinned Ray package; does not start Ray")
    def test_real_ray_cli_accepts_both_heads_without_worker_port_conflicts(self):
        import ray
        from ray._private.parameter import RayParams
        from ray.scripts.scripts import start

        if ray.__version__ != "2.58.0":
            self.skipTest("Requires the example's Ray 2.58.0")
        for directory, port, gpus in ((self.example.server, self.example.base + 2, 8),
                                       (self.example.client, self.example.base + 12, 0)):
            with self.subTest(head_port=port):
                command = self.example.ray_command(directory, port, "tmp-ray", gpus)
                # Parse with the actual CLI, including its defaults. Never invoke
                # the callback, which would start processes or create Ray state.
                with start.make_context("start", list(map(str, command[2:]))) as context:
                    options = context.params
                params = RayParams(
                    redis_port=options["port"],
                    node_manager_port=options["node_manager_port"],
                    object_manager_port=options["object_manager_port"],
                    min_worker_port=options["min_worker_port"],
                    max_worker_port=options["max_worker_port"],
                )
                params.update_pre_selected_port()

    def test_busy_gpu_prevents_any_service_start(self):
        self.fake.busy_gpu = True
        with patch.object(self.example, "prepare_assets"):
            with self.assertRaisesRegex(launch.SetupError, "GPU compute processes"):
                self.example.run()
        self.assertFalse(self.fake.spawns)

    def test_read_only_check_on_existing_checkout_does_not_refresh_index(self):
        (self.example.source / ".git").mkdir(parents=True)
        before = sorted(str(path) for path in self.example.work.rglob("*"))
        self.example.check()
        self.assertEqual(before, sorted(str(path) for path in self.example.work.rglob("*")))
        status_calls = [(args, env) for args, env in self.fake.calls if "status" in args]
        self.assertEqual(status_calls[0][1]["GIT_OPTIONAL_LOCKS"], "0")
        self.assertFalse(self.fake.spawns)

    def test_wrong_model_is_not_overwritten(self):
        self.example.model.mkdir(parents=True)
        (self.example.model / "config.json").write_text('{"model_type": "llama"}')
        (self.example.model / "tokenizer_config.json").write_text("{}")
        with self.assertRaisesRegex(launch.SetupError, "does not match"):
            self.example.model_ready()
        self.assertFalse(self.fake.calls)

    def test_tuft_readiness_requires_health_and_expected_model(self):
        def response(request, **kwargs):
            self.assertEqual(request.get_header("X-api-key"), "fixture-key")
            if request.full_url.endswith("healthz"):
                return io.BytesIO(b'{"status":"ok"}')
            return io.BytesIO(b'{"supported_models":[{"model_name":"Qwen/Qwen3-1.7B"}]}')
        self.example.key = "fixture-key"
        # Bypass setUp's instance stub to exercise the actual HTTP parser.
        with patch.object(launch.urllib.request, "urlopen", side_effect=response):
            self.assertTrue(launch.Example.tuft_ready(self.example))
        with patch.object(launch.urllib.request, "urlopen", return_value=io.BytesIO(b'{"status":"not-ready"}')):
            self.assertFalse(launch.Example.tuft_ready(self.example))

    def test_invalid_steps(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                launch.positive(value)


class RuntimeDependencyTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy") and importlib.util.find_spec("aiosqlite"),
                         "Requires SQLAlchemy and aiosqlite; uses an in-memory database")
    def test_real_sqlalchemy_async_sqlite_connection(self):
        if importlib.metadata.version("sqlalchemy") != "2.1.1":
            self.skipTest("Requires the example's SQLAlchemy 2.1.1")
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        async def query():
            engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            try:
                async with engine.connect() as connection:
                    self.assertEqual((await connection.execute(text("SELECT 1"))).scalar_one(), 1)
            finally:
                await engine.dispose()

        asyncio.run(query())


if __name__ == "__main__":
    unittest.main()
