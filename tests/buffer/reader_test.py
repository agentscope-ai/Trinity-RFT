from tests.tools import RayUnittestBaseAsync, get_unittest_dataset_config
from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.reader import READER
from trinity.buffer.reader.file_reader import FileReader, TaskFileReader


@READER.register_module("custom")
class CustomReader(TaskFileReader):
    """A custom reader for testing."""

    def __init__(self, config):
        super().__init__(config)


class TestBufferReader(RayUnittestBaseAsync):
    async def test_buffer_reader_registration(self) -> None:
        config = get_unittest_dataset_config("countdown", "train")
        config.batch_size = 2
        config.storage_type = "custom"
        reader = get_buffer_reader(config)
        self.assertIsInstance(reader, CustomReader)
        tasks = await reader.read()
        self.assertEqual(len(tasks), 2)
        config.storage_type = "file"
        reader = get_buffer_reader(config)
        self.assertIsInstance(reader, FileReader)
        tasks = await reader.read()
        self.assertEqual(len(tasks), 2)
