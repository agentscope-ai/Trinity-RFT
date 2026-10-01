"""Run the real Bash/Python --check entry with fake hardware, never services."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class EntryEnvironmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name).resolve()
        self.root = self.scratch / "tutorial"
        self.example = self.root / "examples/alfworld_lora32"
        self.example.mkdir(parents=True)
        for name in ("run.sh", "launch.py"):
            shutil.copy2(Path(__file__).parent / name, self.example / name)
        self.bin = self.scratch / "bin"
        self.bin.mkdir()
        (self.bin / "python3").symlink_to(sys.executable)
        # Only replace prerequisite probes. All .env loading, path resolution,
        # CLI parsing and read-only checks execute the production entry files.
        for name in ("uv", "git", "redis-server", "wget"):
            script = self.bin / name
            script.write_text("#!/bin/sh\nexit 99\n")
            script.chmod(0o755)
        gpu = self.bin / "nvidia-smi"
        gpu.write_text("#!/bin/sh\n"
                       '[ "${TINKER_API_KEY:-}" != "chapter1-secret" ] || exit 97\n'
                       '[ "${TRINITY_MACHINE_ID:-}" != "full-param" ] || exit 97\n'
                       "printf 'NVIDIA A100-SXM4-80GB, 81920\\n%.0s' 1 2 3 4 5 6 7 8\n")
        gpu.chmod(0o755)
        platform_fixture = self.scratch / "platform-fixture"
        platform_fixture.mkdir()
        (platform_fixture / "sitecustomize.py").write_text(
            'import platform\nplatform.system = lambda: "Linux"\nplatform.machine = lambda: "x86_64"\n')
        self.env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.scratch),
                    "PYTHONPATH": str(platform_fixture), "PYTHONDONTWRITEBYTECODE": "1",
                    "LORA32_WORK_DIR": str(self.scratch / "work")}

    def invoke(self, args=("--check",), env=None, success=True):
        before = sorted(str(path) for path in self.root.rglob("*"))
        result = subprocess.run(["bash", str(self.example / "run.sh"), *args],
                                cwd=self.scratch, env=dict(self.env, **(env or {})),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))
        self.assertFalse((self.scratch / "work").exists())
        return result.stdout + result.stderr

    def assert_assets(self, output, model, data):
        self.assertIn(f"[check] Model: {self.root / model}", output)
        self.assertIn(f"[check] ALFWorld: {self.root / data}", output)

    def test_no_configuration_uses_chapter1_defaults_before_download(self):
        output = self.invoke()
        self.assert_assets(output, "models/Qwen3-1.7B", "alfworld_data")
        self.assertFalse((self.root / "models").exists())
        self.assertFalse((self.root / "alfworld_data").exists())

    def test_empty_lora_paths_inherit_root_shell_paths_only(self):
        (self.root / ".env").write_text(
            'ASSET_PARENT="./shared assets"\n'
            'TRINITY_MODEL_PATH="$ASSET_PARENT/model"\nALFWORLD_DATA="$ASSET_PARENT/data"\n'
            'TRINITY_MACHINE_ID=full-param\nTINKER_API_KEY=chapter1-secret\n'
            'LORA32_WORK_DIR=/must-not-inherit\nLORA32_PORT_BASE=0\n'
            'TRINITY_CHECKPOINT_ROOT_DIR=/must-not-inherit\nRAY_ADDRESS=other-server:1\n')
        (self.example / ".env").write_text('TRINITY_MODEL_PATH=\nALFWORLD_DATA=\n')
        output = self.invoke()
        self.assert_assets(output, "shared assets/model", "shared assets/data")
        self.assertIn(f"[check] Work directory: {self.scratch / 'work'}", output)
        self.assertNotIn("chapter1-secret", output)

    def test_nonempty_lora_paths_override_root_paths(self):
        (self.root / ".env").write_text('TRINITY_MODEL_PATH=old/model\nALFWORLD_DATA=old/data\n')
        (self.example / ".env").write_text('TRINITY_MODEL_PATH=lora/model\nALFWORLD_DATA=lora/data\n')
        self.assert_assets(self.invoke(), "lora/model", "lora/data")

    def test_nonempty_explicit_paths_override_lora_file(self):
        (self.example / ".env").write_text('TRINITY_MODEL_PATH=lora/model\nALFWORLD_DATA=lora/data\n')
        output = self.invoke(env={"TRINITY_MODEL_PATH": "explicit/model", "ALFWORLD_DATA": "explicit/data"})
        self.assert_assets(output, "explicit/model", "explicit/data")

    def test_empty_lora_path_preserves_explicit_path_and_falls_back_per_field(self):
        (self.root / ".env").write_text('TRINITY_MODEL_PATH=root/model\nALFWORLD_DATA=root/data\n')
        (self.example / ".env").write_text('TRINITY_MODEL_PATH=\nALFWORLD_DATA=\n')
        output = self.invoke(env={"TRINITY_MODEL_PATH": "explicit/model"})
        self.assert_assets(output, "explicit/model", "root/data")

    def test_empty_root_paths_use_defaults_and_relative_workdir_uses_repo_root(self):
        (self.root / ".env").write_text('TRINITY_MODEL_PATH=\nALFWORLD_DATA=\n')
        output = self.invoke(env={"LORA32_WORK_DIR": "../relative-work"})
        self.assert_assets(output, "models/Qwen3-1.7B", "alfworld_data")
        self.assertIn(f"[check] Work directory: {self.scratch / 'relative-work'}", output)
        self.assertFalse((self.scratch / "relative-work").exists())

    def test_help_anywhere_skips_both_configuration_files(self):
        marker = self.scratch / "config-was-sourced"
        for path in (self.root / ".env", self.example / ".env"):
            path.write_text(f'touch "{marker}"\nreturn 8\n')
        output = self.invoke(args=("--steps", "150", "--help"))
        self.assertIn("usage:", output)
        self.assertFalse(marker.exists())

    def test_root_config_failure_does_not_print_secret_diagnostics(self):
        (self.root / ".env").write_text('printf "chapter1-secret" >&2\nreturn 9\n')
        output = self.invoke(success=False)
        self.assertIn("Cannot read the repository .env", output)
        self.assertNotIn("chapter1-secret", output)


if __name__ == "__main__":
    unittest.main()
