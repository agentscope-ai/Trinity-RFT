"""SQLAlchemy models for different data."""

from typing import Dict, Optional, Tuple

from sqlalchemy import JSON, Column, DateTime, Float, Integer, LargeBinary, String, func
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base

from trinity.common.experience import Experience
from trinity.utils.log import get_logger

Base = declarative_base()


class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_task = Column(JSON, nullable=False)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(raw_task=dict)


# ============================================================
# Experience Models (meta + blob split)
# ============================================================


class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience metadata."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now())
    task_id = Column(String(64), nullable=True, index=True)
    run_id = Column(Integer, nullable=True, index=True)
    msg_id = Column(String(64), nullable=True, index=True)
    model_version = Column(Integer, nullable=True, index=True)
    reward = Column(Float, nullable=True, index=True)
    consumed = Column(Integer, default=0, index=True)

    def to_experience(self, blob_bytes: bytes) -> Experience:
        """Load the experience from metadata + blob bytes."""
        exp = Experience.deserialize(blob_bytes)
        exp.eid.task = self.task_id
        exp.eid.run = self.run_id
        exp.eid.suffix = self.msg_id
        exp.reward = self.reward
        exp.info["model_version"] = self.model_version
        return exp

    @classmethod
    def from_experience(cls, experience: Experience):
        """Create meta row from experience (blob stored separately)."""
        return cls(
            reward=experience.reward,
            task_id=str(experience.eid.task),
            run_id=experience.eid.run,
            msg_id=str(experience.eid.suffix),
            model_version=experience.info.get("model_version"),
        )


class BlobModel(Base):  # type: ignore
    """Unified blob storage model for all experience types."""

    __abstract__ = True

    id = Column(Integer, primary_key=True)
    experience_bytes = Column(LargeBinary, nullable=False)


# ============================================================
# SFT Models (meta + blob split)
# ============================================================


class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data metadata."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_list = Column(JSON, nullable=True)

    def to_experience(self, blob_bytes: bytes) -> Experience:
        """Load the experience from metadata + blob bytes."""
        return Experience.deserialize(blob_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Create meta row from experience (blob stored separately)."""
        return cls(
            message_list=experience.messages,
        )


# ============================================================
# DPO Models (meta + blob split)
# ============================================================


class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data metadata."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    chosen_message_list = Column(JSON, nullable=True)
    rejected_message_list = Column(JSON, nullable=True)

    def to_experience(self, blob_bytes: bytes) -> Experience:
        """Load the experience from metadata + blob bytes."""
        return Experience.deserialize(blob_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Create meta row from experience (blob stored separately)."""
        return cls(
            chosen_message_list=experience.chosen_messages,
            rejected_message_list=experience.rejected_messages,
        )


# ============================================================
# Engine initialization
# ============================================================


def _create_table_classes(table_name: str, schema_type: str):
    """Create dynamic table model classes for the given schema type.

    Returns:
        For task schema: (table_cls,)
        For experience/sft/dpo schema: (meta_cls, blob_cls)
    """
    from trinity.buffer.schema import SQL_SCHEMA

    if schema_type is None:
        schema_type = "task"

    base_class = SQL_SCHEMA.get(schema_type)

    if schema_type == "task":
        table_attrs = {
            "__tablename__": table_name,
            "__abstract__": False,
            "__table_args__": {"keep_existing": True},
        }
        table_cls = type(table_name, (base_class,), table_attrs)
        return (table_cls,)

    meta_attrs = {
        "__tablename__": table_name,
        "__abstract__": False,
        "__table_args__": {"keep_existing": True},
    }
    meta_cls = type(f"{table_name}_meta", (base_class,), meta_attrs)

    blob_table_name = f"{table_name}_blob"
    blob_attrs = {
        "__tablename__": blob_table_name,
        "__abstract__": False,
        "__table_args__": {"keep_existing": True},
    }
    blob_cls = type(f"{table_name}_blob", (BlobModel,), blob_attrs)
    return (meta_cls, blob_cls)


async def init_async_engine(db_url: str, table_name: str, schema_type: Optional[str]) -> Tuple:
    """Create an async SQLAlchemy engine and table classes.

    Returns:
        For task schema: (async_engine, table_cls)
        For experience/sft/dpo schema: (async_engine, meta_cls, blob_cls)
    """
    from trinity.buffer.utils import to_async_url

    logger = get_logger(__name__)
    async_url = to_async_url(db_url)
    engine = create_async_engine(async_url, pool_pre_ping=True)

    if schema_type is None:
        schema_type = "task"

    classes = _create_table_classes(table_name, schema_type)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Created async tables for {table_name} (schema={schema_type}).")
    except OperationalError:
        logger.warning(f"Failed to create async tables for {table_name}, assuming they exist.")

    if schema_type == "task":
        return engine, classes[0]
    return engine, classes[0], classes[1]
