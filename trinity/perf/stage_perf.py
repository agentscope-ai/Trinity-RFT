from __future__ import annotations

import json
import os
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ray

from trinity.buffer.pipelines.task_pipeline import check_and_run_task_pipeline
from trinity.common.config import Config, load_config
from trinity.perf.report_metrics import compute_global_token_throughput_metrics
from trinity.perf.resource_sampler import ResourceSampler
from trinity.perf.tensorboard_metrics import (
    TensorBoardScalarReader,
    collect_step_metrics,
)
from trinity.utils.plugin_loader import load_plugins


@dataclass(slots=True)
class ExplorerPerfOptions:
    config_path: str
    output_path: str
    monitor_interval: float = 2.0
    total_steps: int = 5
    timeout: Optional[float] = None


def validate_explorer_perf_config(config: Config) -> None:
    """Validate perf-specific config constraints."""
    if config.mode != "explore":
        raise ValueError(f"Explorer perf requires mode 'explore', got '{config.mode}'.")


def build_explorer_perf_payload(
    *,
    config: Optional[Config],
    options: ExplorerPerfOptions,
    startup_time_sec: Optional[float],
    execution_time_sec: Optional[float],
    total_time_sec: Optional[float],
    resource_payload: dict[str, Any],
    step_metrics: list[dict[str, Any]],
    success: bool,
    error: Optional[str],
) -> dict[str, Any]:
    """Assemble the final JSON payload."""
    artifacts = {}
    explorer_name = None
    if config is not None:
        explorer_name = config.explorer.name
        artifacts = {
            "checkpoint_job_dir": config.checkpoint_job_dir,
            "tensorboard_dir": os.path.join(
                config.monitor.cache_dir, "tensorboard", config.explorer.name
            ),
            "log_dir": config.log.save_dir,
            "output_json": options.output_path,
        }

    timing = {
        "startup_time_sec": startup_time_sec,
        "execution_time_sec": execution_time_sec,
        "total_time_sec": total_time_sec,
    }
    timing.update(
        compute_global_token_throughput_metrics(
            execution_time_sec=execution_time_sec,
            step_metrics=step_metrics,
        )
    )

    return {
        "run_meta": {
            "module": "explorer",
            "config_path": options.config_path,
            "explorer_name": explorer_name,
            "monitor_interval_sec": options.monitor_interval,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "generated_at": time.time(),
        },
        "timing": timing,
        **resource_payload,
        "step_metrics": step_metrics,
        "artifacts": artifacts,
        "status": {
            "success": success,
            "error": error,
            "gpu_metrics_available": bool(resource_payload.get("resource_timeline")),
            "tensorboard_metrics_available": bool(step_metrics),
        },
    }


def write_explorer_perf_output(output_path: str, payload: dict[str, Any]) -> None:
    """Write the final payload to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_explorer_perf(options: ExplorerPerfOptions) -> dict[str, Any]:
    """Run Explorer perf collection and return the result payload."""
    from trinity.cli.launcher import explore

    load_plugins()
    config: Optional[Config] = None
    sampler: Optional[ResourceSampler] = None
    error: Optional[str] = None
    startup_time_sec: Optional[float] = None
    execution_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    resource_payload: dict[str, Any] = {"resource_timeline": []}
    step_metrics: list[dict[str, Any]] = []

    try:
        config = load_config(options.config_path)
        config.buffer.total_steps = options.total_steps
        config.monitor.monitor_type = "tensorboard"
        config.continue_from_checkpoint = False  # ensure we start fresh for perf testing
        validate_explorer_perf_config(config)
        config.check_and_update()

        ray.init(
            address=config.cluster.ray_address,
            ignore_reinit_error=True,
            namespace=config.ray_namespace,
            runtime_env={"env_vars": config.get_envs()},
        )
        check_and_run_task_pipeline(config)

        sampler = ResourceSampler(interval_seconds=options.monitor_interval)
        sampler.start()

        stage_status = explore(config, timeout=options.timeout)
        startup_time_sec = stage_status.startup_time_sec
        execution_time_sec = stage_status.execution_time_sec
        total_time_sec = stage_status.total_time_sec
        if stage_status.error is not None:
            error = stage_status.error.traceback_text
    except (RuntimeError, ValueError) as e:
        error = traceback.format_exc()
        print(f"Explorer perf failed with error: {e}\n{error}")
        raise e
    finally:
        collected_samples = sampler.stop() if sampler is not None else []
        resource_payload = {"resource_timeline": [sample.to_dict() for sample in collected_samples]}

        if config is not None:
            tensorboard_dir = os.path.join(
                config.monitor.cache_dir, "tensorboard", config.explorer.name
            )
            if os.path.isdir(tensorboard_dir):
                scalar_reader = TensorBoardScalarReader(tensorboard_dir)
                step_metrics = collect_step_metrics(scalar_reader.metrics)

        if ray.is_initialized():
            ray.shutdown()

    return build_explorer_perf_payload(
        config=config,
        options=options,
        startup_time_sec=startup_time_sec,
        execution_time_sec=execution_time_sec,
        total_time_sec=total_time_sec,
        resource_payload=resource_payload,
        step_metrics=step_metrics,
        success=error is None,
        error=error,
    )
