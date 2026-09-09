from trinity.utils.registry import Registry

READER = Registry(
    "reader",
    default_mapping={
        "file": "trinity.buffer.reader.file_reader.FileReader",
        "queue": "trinity.buffer.reader.queue_reader.QueueReader",
        "sql": "trinity.buffer.reader.sql_reader.SQLReader",
        "task_dir": "trinity.buffer.reader.task_dir_reader.TaskDirReader",
    },
)

__all__ = ["READER"]
