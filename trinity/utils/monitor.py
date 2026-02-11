"""Monitor"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import swanlab
except ImportError:
    swanlab = None

from torch.utils.tensorboard import SummaryWriter

from trinity.common.config import Config
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

MONITOR = Registry(
    "monitor",
    default_mapping={
        "tensorboard": "trinity.utils.monitor.TensorboardMonitor",
        "wandb": "trinity.utils.monitor.WandbMonitor",
        "mlflow": "trinity.utils.monitor.MlflowMonitor",
        "swanlab": "trinity.utils.monitor.SwanlabMonitor",
    },
)


def gather_metrics(
    metric_list: List[Dict], prefix: str, output_stats: List[str] = ["mean", "max", "min"]
) -> Dict:
    if not metric_list:
        return {}
    try:
        df = pd.DataFrame(metric_list)
        numeric_df = df.select_dtypes(include=[np.number])
        stats_df = numeric_df.agg(output_stats)
        metric = {}
        for col in stats_df.columns:
            for stats in output_stats:
                metric[f"{prefix}/{col}/{stats}"] = stats_df.loc[stats, col].item()
        return metric
    except Exception as e:
        raise ValueError(f"Failed to gather metrics: {e}") from e


def gather_eval_metrics(
    metric_list: List[Dict],
    prefix: str,
    output_stats: List[str] = ["mean", "max", "min", "std"],
    detailed_stats: bool = False,
) -> Dict:
    if not metric_list:
        return {}
    try:
        df = pd.DataFrame(metric_list)
        numeric_df = df.select_dtypes(include=[np.number])
        metric = {}
        for col in numeric_df.columns:
            if detailed_stats:
                stats_df = numeric_df[[col]].agg(output_stats)
                for stats in output_stats:
                    metric[f"{prefix}/{col}/{stats}"] = stats_df.loc[stats, col].item()
            else:
                # only return the mean of the column
                metric[f"{prefix}/{col}"] = numeric_df[col].mean()

        return metric
    except Exception as e:
        raise ValueError(f"Failed to gather eval metrics: {e}") from e


class Monitor(ABC):
    """Monitor"""

    def __init__(
        self,
        project: str,
        name: str,
        role: str,
        config: Config = None,  # pass the global Config for recording
    ) -> None:
        self.project = project
        self.name = name
        self.role = role
        self.config = config

    @abstractmethod
    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        """Log a table"""

    @abstractmethod
    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""

    @abstractmethod
    def close(self) -> None:
        """Close the monitor"""

    def __del__(self) -> None:
        self.close()

    def calculate_metrics(
        self, data: dict[str, Union[List[float], float]], prefix: Optional[str] = None
    ) -> dict[str, float]:
        metrics = {}
        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"

            if isinstance(val, List):
                if len(val) > 1:
                    metrics[f"{key}/mean"] = np.mean(val)
                    metrics[f"{key}/max"] = np.amax(val)
                    metrics[f"{key}/min"] = np.amin(val)
                elif len(val) == 1:
                    metrics[key] = val[0]
            else:
                metrics[key] = val
        return metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}


class TensorboardMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.close()


class WandbMonitor(Monitor):
    """Monitor with Weights & Biases.

    Args:
        base_url (`Optional[str]`): The base URL of the W&B server. If not provided, use the environment variable `WANDB_BASE_URL`.
        api_key (`Optional[str]`): The API key for W&B. If not provided, use the environment variable `WANDB_API_KEY`.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert wandb is not None, "wandb is not installed. Please install it to use WandbMonitor."
        if not group:
            group = name
        monitor_args = config.monitor.monitor_args or {}
        if base_url := monitor_args.get("base_url"):
            os.environ["WANDB_BASE_URL"] = base_url
        if api_key := monitor_args.get("api_key"):
            os.environ["WANDB_API_KEY"] = api_key
        self.logger = wandb.init(
            project=project,
            group=group,
            name=f"{name}_{role}",
            tags=[role],
            config=config,
            save_code=False,
        )
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table = wandb.Table(dataframe=experiences_table)
        self.log(data={table_name: experiences_table}, step=step)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        self.logger.log(data, step=step, commit=commit)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.finish()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "base_url": None,
            "api_key": None,
        }


class MlflowMonitor(Monitor):
    """Monitor with MLflow.

    Args:
        uri (`Optional[str]`): The tracking server URI. If not provided, the default is `http://localhost:5000`.
        username (`Optional[str]`): The username to login. If not provided, the default is `None`.
        password (`Optional[str]`): The password to login. If not provided, the default is `None`.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert (
            mlflow is not None
        ), "mlflow is not installed. Please install it to use MlflowMonitor."
        monitor_args = config.monitor.monitor_args or {}
        if username := monitor_args.get("username"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password := monitor_args.get("password"):
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        mlflow.set_tracking_uri(config.monitor.monitor_args.get("uri", "http://localhost:5000"))
        mlflow.set_experiment(project)
        mlflow.enable_system_metrics_logging()
        mlflow.start_run(
            run_name=f"{name}_{role}",
            tags={
                "group": group,
                "role": role,
            },
        )
        mlflow.log_params(config.flatten())
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table["step"] = step
        mlflow.log_table(data=experiences_table, artifact_file=f"{table_name}.json")

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        self.console_logger.info(f"Step {step}: {data}")
        # Replace all '@' in keys with '_at_', as MLflow does not support '@' in metric names
        data = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=data, step=step)

    def close(self) -> None:
        mlflow.end_run()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "uri": "http://localhost:5000",
            "username": None,
            "password": None,
        }


class SwanlabMonitor(Monitor):
    """Monitor with SwanLab (https://swanlab.cn/).

    Set `SWANLAB_API_KEY` environment variable with your SwanLab API key before using this monitor.
    If you're using local deployment of Swanlab, also set `SWANLAB_API_HOST` environment variable.
    Pass additional SwanLab initialization arguments via `config.monitor.monitor_args` in the Config,
    such as `tags`, `description`, `logdir`, etc. See SwanLab documentation for details.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert (
            swanlab is not None
        ), "swanlab is not installed. Please install it to use SwanlabMonitor."

        monitor_args = config.monitor.monitor_args or {}

        # Optional API login via code if provided; otherwise try environment, then rely on prior `swanlab login`.
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            try:
                swanlab.login(api_key=api_key, save=True)
            except Exception:
                # Best-effort login; continue to init which may still work if already logged in
                pass
        else:
            raise RuntimeError("SWANLAB_API_KEY environment variable not set.")

        # Compose tags (ensure list and include role/group markers)
        tags = monitor_args.get("tags") or []
        if isinstance(tags, tuple):
            tags = list(tags)
        if role and role not in tags:
            tags.append(role)
        if group and group not in tags:
            tags.append(group)

        # Determine experiment name
        exp_name = monitor_args.get("experiment_name") or f"{name}_{role}"
        self.exp_name = exp_name

        # Prepare init kwargs, passing only non-None values to respect library defaults
        init_kwargs = {
            "project": project,
            "workspace": monitor_args.get("workspace"),
            "experiment_name": exp_name,
            "description": monitor_args.get("description"),
            "tags": tags or None,
            "logdir": monitor_args.get("logdir"),
            "mode": monitor_args.get("mode") or "cloud",
            "settings": monitor_args.get("settings"),
            "id": monitor_args.get("id"),
            "config": config.flatten() if config is not None else None,
            "resume": monitor_args.get("resume"),
            "reinit": monitor_args.get("reinit"),
        }
        # Strip None values to avoid overriding swanlab defaults
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        self.logger = swanlab.init(**init_kwargs)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        # Not support log table yet
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        # SwanLab doesn't use commit flag; keep signature for compatibility
        swanlab.log(data, step=step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        try:
            # Prefer run.finish() if available
            if hasattr(self, "logger") and hasattr(self.logger, "finish"):
                self.logger.finish()
            else:
                # Fallback to global finish
                swanlab.finish()
        except Exception as e:
            self.console_logger.warning(f"Failed to close SwanlabMonitor: {e}")

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}
