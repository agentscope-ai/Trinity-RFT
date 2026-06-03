from trinity.buffer.schema.sql_schema import init_async_engine
from trinity.utils.registry import Registry

FORMATTER: Registry = Registry(
    "formatter",
    {
        "task": "trinity.buffer.schema.formatter.TaskFormatter",
        "sft": "trinity.buffer.schema.formatter.SFTFormatter",
        "dpo": "trinity.buffer.schema.formatter.DPOFormatter",
    },
)

SQL_SCHEMA: Registry = Registry(
    "sql_schema",
    {
        "task": "trinity.buffer.schema.sql_schema.TaskModel",
        "experience": "trinity.buffer.schema.sql_schema.ExperienceModel",
        "sft": "trinity.buffer.schema.sql_schema.SFTDataModel",
        "dpo": "trinity.buffer.schema.sql_schema.DPODataModel",
    },
)

__all__ = ["init_async_engine", "FORMATTER", "SQL_SCHEMA"]
