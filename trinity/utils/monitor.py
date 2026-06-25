"""Monitor"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional

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

_logger = get_logger(__name__)


class Monitor(ABC):
    """Monitor"""

    # whether this backend can merge explorer+trainer into one shared run
    supports_shared_run: bool = False

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

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        """Log a table"""
        pass

    @abstractmethod
    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""

    @abstractmethod
    def close(self) -> None:
        """Close the monitor"""

    def __del__(self) -> None:
        self.close()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}

    # ---- shared-run hooks (launcher/primary side) -------------------------------
    # Default implementations make a backend "shared-run unaware": init_shared declines
    # (-> separate runs), poll_shared is a no-op. Backends opt in by overriding.

    @classmethod
    def init_shared(cls, config: Config) -> Optional[object]:
        """Create the primary shared run; default declines (unsupported backend).

        Args:
            config: Global config; the run id is written back to ``config.monitor.run_id``.

        Returns:
            The primary run handle, or ``None`` if the backend can't share runs.
        """
        _logger.warning(
            "monitor `%s` does not support shared_run; falling back to separate runs.",
            config.monitor.monitor_type,
        )
        return None

    @classmethod
    def poll_shared(cls, handle, config: Config, declared: set) -> None:
        """Re-declare newly-seen bindings on the primary; default no-op.

        Args:
            handle: The primary run.
            config: Global config.
            declared: Keys already declared on the primary (mutated to add new ones).
        """

    @staticmethod
    def finish_shared(handle) -> None:
        """Finish the primary shared run, best effort.

        Args:
            handle: The primary run to finish, or ``None`` to skip.
        """
        if handle is None:
            return
        try:
            handle.finish()
        except Exception:
            _logger.warning("Failed to finish shared run", exc_info=True)

    @staticmethod
    def load_shared_step_bindings(cache_dir: str) -> Dict[str, str]:
        """Merge all actors' binding files into one mapping.

        Args:
            cache_dir: Monitor cache dir holding ``shared_step_bindings/*.json``.

        Returns:
            Merged ``metric key -> step axis`` mapping.
        """
        bindings: Dict[str, str] = {}
        if not cache_dir:
            return bindings
        d = os.path.join(cache_dir, "shared_step_bindings")
        if not os.path.isdir(d):
            return bindings
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn)) as f:
                    bindings.update(json.load(f))
            except Exception:
                _logger.warning("Failed to read bindings file %s", fn, exc_info=True)
        return bindings

    @staticmethod
    def apply_shared_step_bindings(handle, bindings: Dict[str, str]) -> None:
        """Declare each ``metric -> step axis`` binding on the primary run.

        Args:
            handle: The primary run.
            bindings: ``metric key -> step axis`` mapping to declare.
        """
        if handle is None or not bindings:
            return
        for key, step_metric in bindings.items():
            try:
                handle.define_metric(key, step_metric=step_metric)
            except Exception:
                _logger.warning("Failed to bind %s -> %s", key, step_metric, exc_info=True)

    def format_data_str(self, data: dict, step: int) -> str:
        cleaned_data = {
            k: (
                v.item()
                if hasattr(v, "item")
                else float(v)  # tensor or numpy scalar
                if isinstance(v, (np.integer, np.floating))
                else v  # numpy types
            )
            for k, v in data.items()
        }
        # Format floats to reasonable precision using default str (avoids scientific notation and long decimals)
        formatted_data = (
            "{"
            + ", ".join(
                repr(k) + ": " + (f"{v:.6g}" if isinstance(v, float) else repr(v))
                for k, v in cleaned_data.items()
            )
            + "}"
        )
        return f"Step {step}: {formatted_data}"


class TensorboardMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"{self.format_data_str(data, step)}")

    def close(self) -> None:
        self.logger.close()


class WandbMonitor(Monitor):
    """Monitor with Weights & Biases.

    Args:
        base_url (`Optional[str]`): The base URL of the W&B server. If not provided, use the environment variable `WANDB_BASE_URL`.
        api_key (`Optional[str]`): The API key for W&B. If not provided, use the environment variable `WANDB_API_KEY`.
    """

    supports_shared_run: bool = True

    @staticmethod
    def _apply_credentials(monitor_args: dict) -> None:
        """Export ``base_url``/``api_key`` from monitor_args into wandb env vars.

        Args:
            monitor_args: Monitor args optionally holding ``base_url``/``api_key``.
        """
        if base_url := monitor_args.get("base_url"):
            os.environ["WANDB_BASE_URL"] = base_url
        if api_key := monitor_args.get("api_key"):
            os.environ["WANDB_API_KEY"] = api_key

    @classmethod
    def init_shared(cls, config: Config) -> Optional[object]:
        """Create the primary run (x_primary) and inject ``config.monitor.run_id``.

        Args:
            config: Global config; the run id is written back to ``config.monitor.run_id``.

        Returns:
            The primary run, or ``None`` in offline mode (no server-side merge).
        """
        assert wandb is not None, "wandb is not installed. Please install it to use WandbMonitor."
        cls._apply_credentials(config.monitor.monitor_args or {})
        if os.environ.get("WANDB_MODE") == "offline":
            _logger.warning("wandb offline mode does not support shared runs; using separate runs.")
            return None
        run = wandb.init(
            project=config.project,
            group=config.group or config.name,
            name=config.name,
            config=config,
            save_code=False,
            settings=wandb.Settings(mode="shared", x_primary=True),
        )
        config.monitor.run_id = run.id
        cls._define_shared_step_axes(run, config)
        return run

    @staticmethod
    def _define_shared_step_axes(run, config: Config) -> None:
        """Declare each role's hidden ``{role}/step`` axis on the primary run.

        Args:
            run: The primary run.
            config: Global config providing the explorer/trainer role names.
        """
        # hidden: each axis is an x-axis only, with no plot of its own
        for role in (config.explorer.name, config.trainer.name):
            run.define_metric(f"{role}/step", hidden=True)

    @classmethod
    def poll_shared(cls, handle, config: Config, declared: set) -> None:
        """Declare any binding keys not yet in ``declared`` on the primary.

        Args:
            handle: The primary run.
            config: Global config (provides ``cache_dir``).
            declared: Keys already declared; newly-seen keys are added in place.
        """
        bindings = cls.load_shared_step_bindings(config.monitor.cache_dir)
        new = {k: v for k, v in bindings.items() if k not in declared}
        if not new:
            return
        # define_metric is retroactive: already-written history rows realign once declared
        cls.apply_shared_step_bindings(handle, new)
        declared.update(new)

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert wandb is not None, "wandb is not installed. Please install it to use WandbMonitor."
        if not group:
            group = name
        self.console_logger = get_logger(__name__, in_ray_actor=True)
        # init these before wandb.init so a failed init still leaves close()/__del__ safe
        # (Option A: shared_writer is created only after a successful shared init below)
        self.logger = None
        self.shared_writer: Optional[SharedRunWriter] = None
        self._apply_credentials(config.monitor.monitor_args or {})
        # join the launcher-created primary run as a secondary writer when enabled
        self.shared = bool(config.monitor.shared_run and config.monitor.run_id)
        if self.shared:
            self.logger = wandb.init(
                project=project,
                group=group,
                id=config.monitor.run_id,
                config=config,
                resume="allow",
                reinit=True,
                save_code=False,
                settings=wandb.Settings(
                    mode="shared",
                    x_primary=False,
                    x_update_finish_state=False,  # this writer's close must not finish the run
                ),
            )
            self.shared_writer = SharedRunWriter(
                self.logger,
                role,
                getattr(config.monitor, "cache_dir", "") or "",
                self.console_logger,
                config,
            )
        else:
            self.logger = wandb.init(
                project=project,
                group=group,
                name=f"{name}_{role}",
                tags=[role],
                config=config,
                save_code=False,
            )

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table = wandb.Table(dataframe=experiences_table)
        self.log(data={table_name: experiences_table}, step=step)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        if self.shared:
            self.shared_writer.record(data, step)
        else:
            self.logger.log(data, step=step, commit=commit)
        self.console_logger.info(f"{self.format_data_str(data, step)}")

    def close(self) -> None:
        if self.shared_writer is not None:
            self.shared_writer.flush()
        if self.logger is not None:
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
        self.console_logger.info(f"{self.format_data_str(data, step)}")
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

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        # SwanLab doesn't use commit flag; keep signature for compatibility
        swanlab.log(data, step=step)
        self.console_logger.info(f"{self.format_data_str(data, step)}")

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


class SharedRun:
    """Launcher-side context manager that creates and finishes a shared run.

    A no-op unless ``config.monitor.shared_run`` is on and the backend supports it.
    The secondaries keep the live step-axis bindings aligned (each carries the full set),
    so the launcher only declares them once on the primary at exit -- which is what makes
    the x-axis correct after finalize (the primary's config wins when the run finalizes).
    """

    def __init__(self, config: Config) -> None:
        """Resolve the backend (no run created yet).

        Args:
            config: Global config selecting the monitor backend.
        """
        self.config = config
        self.handle = None
        self._cls = MONITOR.get(config.monitor.monitor_type)
        self._declared: set = set()

    def __enter__(self) -> "SharedRun":
        """Create the primary run.

        Returns:
            Self, for use as a context manager.
        """
        if not self.config.monitor.shared_run or self._cls is None:
            return self
        self.handle = self._cls.init_shared(self.config)  # create primary, inject run_id
        return self

    def __exit__(self, *exc) -> Literal[False]:
        """Declare the bindings once on the primary, then finish the run.

        Args:
            *exc: Exception triple from the ``with`` block (unused).

        Returns:
            ``False``, so exceptions from the with-body are never suppressed.
        """
        if self.handle is not None:
            try:
                # one-shot: declare all bindings on the primary before finalize
                self._cls.poll_shared(self.handle, self.config, self._declared)
            except Exception:
                _logger.warning("shared-run poll failed", exc_info=True)
            self._cls.finish_shared(self.handle)
        return False  # never suppress exceptions from the with-body


class SharedRunWriter:
    """Secondary-side writer: buffers one row per step and persists step-axis bindings.

    Created only after a successful ``wandb.init`` (Option A), so ``run`` is always valid.
    """

    def __init__(self, run, role: str, cache_dir: str, console_logger, config: Config) -> None:
        """Bind the shared run and init the per-step buffer / binding state.

        Args:
            run: The joined secondary wandb run.
            role: This actor's role, used as the ``{role}/`` metric prefix and step axis.
            cache_dir: Monitor cache dir; ``""`` disables binding persistence.
            console_logger: Logger for best-effort persistence warnings.
            config: Global config, for the explorer/trainer role names.
        """
        self._run = run
        self.role = role
        self._console_logger = console_logger
        self._cache_dir = cache_dir
        # buffer one step's metrics, emitted as a single row when the step advances
        self._buffer: dict = {}
        self._buffer_step = None
        # full keys seen so far, so the primary binds each to the step axis exactly once
        self._logged_keys: set = set()
        # keys already define_metric'd on THIS secondary handle (they persist in its config)
        self._declared: set = set()
        # cache file the launcher reads at finish; None disables persistence
        self._bindings_path = (
            os.path.join(cache_dir, "shared_step_bindings", f"{role}.json") if cache_dir else None
        )
        self._persisted_count = 0
        # Declare BOTH roles' hidden step axes on this secondary so its config matches the
        # primary's and its syncs don't drop the merged step-axis defs.
        for r in (config.explorer.name, config.trainer.name):
            self._run.define_metric(f"{r}/step", hidden=True)

    def record(self, data: dict, step: int) -> None:
        """Buffer a step's metrics, flushing the previous step when ``step`` advances.

        Args:
            data: Metric name -> value for this step.
            step: This role's step counter.
        """
        if self._buffer_step is not None and step != self._buffer_step:
            self.flush()
        self._buffer_step = step
        self._buffer.update({f"{self.role}/{k}": v for k, v in data.items()})

    def flush(self) -> None:
        """Emit the buffered step as one wandb history row."""
        if self._buffer_step is None:
            return
        self._logged_keys.update(self._buffer.keys())
        # Declare the FULL binding set on this secondary handle, not just this role's keys:
        # in wandb shared mode a writer's config sync replaces the merged metric defs, so a
        # secondary holding only its own role's bindings drops the other role's on every
        # sync -- the two secondaries clobber each other and only one role's x-axis is
        # correct at a time. Every role's published bindings + this step's own keys make
        # this secondary carry the whole set, so no sync drops anything (verified: both
        # roles stay bound throughout the run, no flicker).
        full = dict(Monitor.load_shared_step_bindings(self._cache_dir))
        full.update({k: f"{self.role}/step" for k in self._buffer})
        for key, step_metric in full.items():
            if key not in self._declared:
                try:
                    self._run.define_metric(key, step_metric=step_metric)
                except Exception:
                    self._console_logger.warning("Failed to bind %s", key, exc_info=True)
                self._declared.add(key)
        # inject the axis value only at log time so it never enters the buffer
        self._run.log({**self._buffer, f"{self.role}/step": self._buffer_step})
        self._buffer = {}
        self._buffer_step = None
        self._persist_bindings()

    def step_bindings(self) -> Dict[str, str]:
        """Return this role's bindings for the primary to declare.

        Returns:
            ``metric key -> {role}/step`` for every key this role has logged.
        """
        # include any buffered-but-not-yet-flushed keys so none are missed
        self._logged_keys.update(self._buffer.keys())
        return {key: f"{self.role}/step" for key in self._logged_keys}

    def _persist_bindings(self) -> None:
        """Atomically write the bindings to the cache file for the launcher."""
        # nothing to do without a cache path, or when no new keys appeared since last write
        if not self._bindings_path or len(self._logged_keys) == self._persisted_count:
            return
        try:
            os.makedirs(os.path.dirname(self._bindings_path), exist_ok=True)
            tmp = f"{self._bindings_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(self.step_bindings(), f)
            os.replace(tmp, self._bindings_path)
            self._persisted_count = len(self._logged_keys)
        except Exception:
            self._console_logger.warning("Failed to persist step bindings", exc_info=True)
