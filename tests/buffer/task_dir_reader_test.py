import shutil
import unittest
from pathlib import Path

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.reader.task_dir_reader import TaskDirReader
from trinity.common.config import TasksetConfig
from trinity.common.constants import StorageType


class TestTaskDirReader(unittest.IsolatedAsyncioTestCase):
    temp_dir = Path("tmp/test_task_dir_reader")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        cls.temp_dir.mkdir(parents=True)
        cls._write_task_dir("task-a")
        cls._write_task_dir("task-b")
        cls._write_task_dir(".ignored")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _write_task_dir(cls, dirname: str) -> None:
        task_dir = cls.temp_dir / dirname
        task_dir.mkdir()
        (task_dir / "payload.txt").write_text(dirname)

    def _config(self, *, index: int = 0, total_epochs: int = 1) -> TasksetConfig:
        config = TasksetConfig(
            name="folder_tasks",
            storage_type=StorageType.TASK_DIR.value,
            path=str(self.temp_dir),
            default_workflow_type="simple_workflow",
            batch_size=1,
            index=index,
            total_epochs=total_epochs,
        )
        config.data_selector = None
        return config

    async def test_read_task_dirs(self):
        reader = get_buffer_reader(self._config())

        self.assertIsInstance(reader, TaskDirReader)
        self.assertEqual(len(reader), 2)

        tasks = await reader.read(batch_size=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].raw_task["task_name"], "task-a")
        self.assertEqual(tasks[0].raw_task["taskset_name"], "folder_tasks")
        self.assertEqual(tasks[0].raw_task["source_type"], "task_dir")
        self.assertTrue(Path(tasks[0].raw_task["task_dir"]).is_absolute())
        self.assertEqual(tasks[0].index["index"], 0)
        self.assertEqual(tasks[1].index["index"], 1)

    async def test_index_file_limits_and_orders_task_dirs(self):
        index_dir = Path("tmp/test_task_dir_reader_index")
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True)
        for dirname in ["task-a", "task-b", "task-c"]:
            task_dir = index_dir / dirname
            task_dir.mkdir()
            (task_dir / "payload.txt").write_text(dirname)
        (index_dir / TaskDirReader.INDEX_FILENAME).write_text(
            "# one task folder per line\n" "task-b\n" "\n" "task-a\n"
        )

        try:
            config = TasksetConfig(
                name="indexed_folder_tasks",
                storage_type=StorageType.TASK_DIR.value,
                path=str(index_dir),
                default_workflow_type="simple_workflow",
                batch_size=1,
                total_epochs=1,
            )
            config.data_selector = None
            reader = get_buffer_reader(config)

            self.assertEqual(len(reader), 2)
            tasks = await reader.read(batch_size=2)
            self.assertEqual(
                [task.raw_task["task_name"] for task in tasks],
                ["task-b", "task-a"],
            )
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)

    async def test_resume_offset(self):
        reader = get_buffer_reader(self._config(index=1, total_epochs=2))
        tasks = await reader.read(batch_size=2)

        self.assertEqual(
            [task.raw_task["task_name"] for task in tasks],
            [
                "task-b",
                "task-a",
            ],
        )
        self.assertEqual(reader.state_dict()["current_index"], 3)


if __name__ == "__main__":
    unittest.main()
