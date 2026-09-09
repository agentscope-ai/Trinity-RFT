import shutil
import unittest
from pathlib import Path

from trinity.common.workflows import HarborWorkflow
from trinity.common.workflows.workflow import Task


class _TestHarborWorkflow(HarborWorkflow):
    def run(self):
        return []


class TestHarborWorkflow(unittest.TestCase):
    temp_dir = Path("tmp/test_harbor_workflow")

    def setUp(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_task(self, name: str = "task-a") -> Path:
        task_dir = self.temp_dir / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("[task]\n" f'name = "test-org/{name}"\n')
        (task_dir / "instruction.md").write_text("Solve the task.")
        return task_dir

    def _task(self, task_dir: Path) -> Task:
        return Task(
            workflow=_TestHarborWorkflow,
            raw_task={
                "task_id": f"test:{task_dir.name}",
                "task_name": task_dir.name,
                "task_dir": str(task_dir),
                "source_type": "task_dir",
            },
        )

    def test_loads_harbor_task_from_task_dir(self):
        task_dir = self._write_task()

        workflow = _TestHarborWorkflow(task=self._task(task_dir), model=None)

        self.assertEqual(workflow.harbor_task_dir, task_dir.resolve())
        self.assertEqual(workflow.harbor_task_name, "task-a")
        self.assertEqual(workflow.harbor_task_config.task.name, "test-org/task-a")
        self.assertEqual(workflow.harbor_instruction, "Solve the task.")
        self.assertTrue(workflow.harbor_task_paths.config_path.exists())
        self.assertTrue(workflow.harbor_task_paths_info["has_config"])
        self.assertTrue(workflow.harbor_task_paths_info["has_instruction"])

    def test_requires_valid_harbor_task_config(self):
        task_dir = self.temp_dir / "missing-config"
        task_dir.mkdir()

        with self.assertRaisesRegex(ValueError, "Failed to load Harbor task config"):
            _TestHarborWorkflow(task=self._task(task_dir), model=None)


if __name__ == "__main__":
    unittest.main()
