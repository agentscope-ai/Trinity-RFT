"""Launch the trainer"""
import contextlib
import os
import time
import traceback
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import ray
import typer
from typing_extensions import Annotated

from trinity.cli.convert import convert_command
from trinity.cli.debug import debug
from trinity.cli.log import log_command
from trinity.cli.perf import perf_app
from trinity.cli.studio import studio_command
from trinity.cli.view import view_command
from trinity.common.config import Config, load_config
from trinity.common.constants import PLUGIN_DIRS_ENV_VAR
from trinity.utils.dlc_utils import is_running, setup_ray_cluster, stop_ray_cluster
from trinity.utils.log import get_logger
from trinity.utils.plugin_loader import load_plugins

logger = get_logger(__name__, in_ray_actor=True)

app = typer.Typer(
    help="Trinity CLI - Launch and manage Trinity-RFT processes.",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


@dataclass(slots=True)
class StageError:
    type_name: str
    message: str
    traceback_text: str


@dataclass(slots=True)
class StageStatus:
    stage: str
    success: bool
    startup_time_sec: Optional[float] = None
    execution_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    error: Optional[StageError] = None


def _build_stage_error(error: BaseException) -> StageError:
    return StageError(
        type_name=type(error).__name__,
        message=str(error),
        traceback_text=traceback.format_exc(),
    )


def bench(config: Config, *, timeout: Optional[float] = None) -> StageStatus:
    """Evaluate model."""
    from trinity.explorer.explorer import Explorer

    config.explorer.name = "benchmark"
    explorer = Explorer.get_actor(config)
    startup_started_at = time.perf_counter()
    startup_time_sec: Optional[float] = None
    try:
        ray.get(explorer.prepare.remote(), timeout=timeout)
        startup_time_sec = time.perf_counter() - startup_started_at

        run_started_at = time.perf_counter()
        ray.get(explorer.benchmark.remote(), timeout=timeout)
        execution_time_sec = time.perf_counter() - run_started_at
        logger.info("Benchmark finished.")
        return StageStatus(
            stage="bench",
            success=True,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
        )
    except Exception as exc:
        error = _build_stage_error(exc)
        logger.error(f"Benchmark failed:\n{error.traceback_text}")
        return StageStatus(
            stage="bench",
            success=False,
            startup_time_sec=startup_time_sec,
            execution_time_sec=None,
            total_time_sec=time.perf_counter() - startup_started_at,
            error=error,
        )
    finally:
        ray.get(explorer.shutdown.remote(), timeout=timeout)


def explore(config: Config, *, timeout: Optional[float] = None) -> StageStatus:
    """Run explorer."""
    from trinity.explorer.explorer import Explorer

    explorer = Explorer.get_actor(config)
    startup_started_at = time.perf_counter()
    startup_time_sec: Optional[float] = None
    run_started_at: Optional[float] = None

    try:
        ray.get(explorer.prepare.remote(), timeout=timeout)
        startup_time_sec = time.perf_counter() - startup_started_at

        run_started_at = time.perf_counter()
        ray.get(explorer.sync_weight.remote(), timeout=timeout)
        ray.get(explorer.explore.remote(), timeout=timeout)
        execution_time_sec = time.perf_counter() - run_started_at
        return StageStatus(
            stage="explore",
            success=True,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
        )
    except Exception as exc:
        error = _build_stage_error(exc)
        logger.error(f"Explorer failed:\n{error.traceback_text}")
        execution_time_sec = (
            time.perf_counter() - run_started_at if run_started_at is not None else None
        )
        return StageStatus(
            stage="explore",
            success=False,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
            error=error,
        )
    finally:
        ray.get(explorer.shutdown.remote(), timeout=timeout)


def train(config: Config, *, timeout: Optional[float] = None) -> StageStatus:
    """Run trainer."""
    from trinity.trainer.trainer import Trainer

    trainer = Trainer.get_actor(config)
    startup_started_at = time.perf_counter()
    startup_time_sec: Optional[float] = None
    run_started_at: Optional[float] = None

    try:
        ray.get(trainer.prepare.remote(), timeout=timeout)
        startup_time_sec = time.perf_counter() - startup_started_at

        run_started_at = time.perf_counter()
        ray.get(trainer.sync_weight.remote(), timeout=timeout)
        ray.get(trainer.train.remote(), timeout=timeout)
        execution_time_sec = time.perf_counter() - run_started_at
        return StageStatus(
            stage="train",
            success=True,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
        )
    except Exception as exc:
        error = _build_stage_error(exc)
        logger.error(f"Trainer failed:\n{error.traceback_text}")
        execution_time_sec = (
            time.perf_counter() - run_started_at if run_started_at is not None else None
        )
        return StageStatus(
            stage="train",
            success=False,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
            error=error,
        )
    finally:
        ray.get(trainer.shutdown.remote(), timeout=timeout)


def serve(config: Config, *, timeout: Optional[float] = None) -> StageStatus:
    """Run explorer in server mode."""
    from trinity.explorer.explorer import Explorer

    explorer = Explorer.get_actor(config)
    startup_started_at = time.perf_counter()
    startup_time_sec: Optional[float] = None
    run_started_at: Optional[float] = None

    try:
        ray.get(explorer.prepare.remote(), timeout=timeout)
        startup_time_sec = time.perf_counter() - startup_started_at

        run_started_at = time.perf_counter()
        ray.get(explorer.sync_weight.remote(), timeout=timeout)
        ray.get(explorer.serve.remote(), timeout=timeout)
        execution_time_sec = time.perf_counter() - run_started_at
        return StageStatus(
            stage="serve",
            success=True,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
        )
    except Exception as exc:
        error = _build_stage_error(exc)
        logger.error(f"Explorer failed:\n{error.traceback_text}")
        execution_time_sec = (
            time.perf_counter() - run_started_at if run_started_at is not None else None
        )
        return StageStatus(
            stage="serve",
            success=False,
            startup_time_sec=startup_time_sec,
            execution_time_sec=execution_time_sec,
            total_time_sec=time.perf_counter() - startup_started_at,
            error=error,
        )
    finally:
        ray.get(explorer.shutdown.remote(), timeout=timeout)


def both(config: Config) -> StageStatus:
    """Setup both explorer and trainer.

    For the explorer, a step contains `batch_size * sync_interval` number
    of rollout tasks.

    For the trainer, it has to consume all experiences generated by the explorer in
    the latest step. The specific number of experiences may vary for different
    algorithms and tasks.
    """
    from trinity.common.constants import SyncMethod
    from trinity.explorer.explorer import Explorer
    from trinity.manager.synchronizer import Synchronizer
    from trinity.trainer.trainer import Trainer
    from trinity.utils.monitor import SharedRun

    # A shared run (one backend run shared by explorer + trainer) is opt-in. When enabled,
    # the orchestration runs inside SharedRun, which creates the shared run on entry and
    # tears it down on exit; when disabled, it runs unwrapped — same behavior as before.
    shared_run_ctx = SharedRun(config) if config.monitor.shared_run else contextlib.nullcontext()
    with shared_run_ctx:
        explorer = Explorer.get_actor(config)
        trainer = Trainer.get_actor(config)
        started_at = time.perf_counter()
        try:
            ray.get([explorer.__ray_ready__.remote(), trainer.__ray_ready__.remote()])
            ray.get(
                [
                    explorer.prepare.remote(),
                    trainer.prepare.remote(),
                ]
            )
            # Set up NCCL weight sync group between Trainer and Explorer.
            # This must happen after both sides are prepared (Trainer has model
            # meta cached, Explorer has rollout models created) and before the
            # first weight sync.
            if config.synchronizer.sync_method == SyncMethod.NCCL:
                synchronizer = Synchronizer.get_actor(namespace=config.ray_namespace)
                ray.get(synchronizer.coordinate_weight_sync_setup.remote())
            ray.get(
                [
                    explorer.sync_weight.remote(),
                    trainer.sync_weight.remote(),
                ]
            )
            ready_ref, wait_ref = ray.wait(
                [
                    explorer.explore.remote(),
                    trainer.train.remote(),
                ],
                num_returns=1,
            )

            ready = ray.get(ready_ref[0])
            if ready == config.trainer.name:
                logger.info(
                    "===========================================================\n"
                    "> Launcher detected that the `Trainer` process has finished.\n"
                    "> Stopping the explorer process immediately.\n"
                    "==========================================================="
                )
                ray.wait(wait_ref, timeout=5)
            elif ready == config.explorer.name:
                logger.info(
                    "===============================================================\n"
                    "> Launcher detected that the `Explorer` process has finished.\n"
                    "> `Trainer` process may need to save the model checkpoint.\n"
                    f"> Waiting {config.synchronizer.sync_timeout} s for the trainer process...\n"
                    "> You can force stop the `Trainer` process by pressing Ctrl+C.\n"
                    "==============================================================="
                )
                ray.wait(wait_ref, timeout=config.synchronizer.sync_timeout)
            return StageStatus(
                stage="both",
                success=True,
                total_time_sec=time.perf_counter() - started_at,
            )
        except Exception as exc:
            error = _build_stage_error(exc)
            logger.error(f"Explorer or Trainer failed:\n{error.traceback_text}")
            return StageStatus(
                stage="both",
                success=False,
                total_time_sec=time.perf_counter() - started_at,
                error=error,
            )
        finally:
            # Tear down the NCCL weight sync group before shutting down actors.
            # Best-effort: if actors or Synchronizer are already dead, skip.
            if config.synchronizer.sync_method == SyncMethod.NCCL:
                try:
                    synchronizer = Synchronizer.get_actor(namespace=config.ray_namespace)
                    ray.get(synchronizer.coordinate_weight_sync_teardown.remote(), timeout=30)
                except Exception:
                    logger.warning("Weight sync teardown skipped (actors may have already exited).")
            ray.wait(
                [explorer.shutdown.remote(), trainer.shutdown.remote()],
                timeout=config.synchronizer.sync_timeout,
                num_returns=2,
            )


MODE_MAP = {
    "explore": explore,
    "train": train,
    "both": both,
    "bench": bench,
    "serve": serve,
    "colocate": both,
}


def run_stage(config: Config) -> StageStatus:
    ray.init(
        address=config.cluster.ray_address,
        ignore_reinit_error=True,
        namespace=config.ray_namespace,
        runtime_env={"env_vars": config.get_envs()},
    )
    pprint(config)
    try:
        from trinity.buffer.pipelines.task_pipeline import check_and_run_task_pipeline

        check_and_run_task_pipeline(config)
        return MODE_MAP[config.mode](config)  # type: ignore[operator]
    finally:
        if config.monitor.enable_ray_timeline:
            timeline_file = os.path.join(config.monitor.cache_dir, "timeline.json")
            logger.info(f"Exporting Ray timeline to {timeline_file}...")
            ray.timeline(filename=timeline_file)
            logger.info("Done. You can open the timeline file in `chrome://tracing`")
        ray.shutdown()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to the config file."),
    ],
    dlc: Annotated[
        bool,
        typer.Option("--dlc", help="Specify when running in Aliyun PAI DLC."),
    ] = False,
    plugin_dir: Annotated[
        Optional[str],
        typer.Option("--plugin-dir", help="Path to the directory containing plugin modules."),
    ] = None,
) -> None:
    """Run RFT process."""
    if plugin_dir:
        os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir
    load_plugins()
    cfg = load_config(config)

    if dlc:
        cluster_namespace = "-".join(p for p in [cfg.project, cfg.group, cfg.name] if p)
        cfg.cluster.ray_address = setup_ray_cluster(namespace=cluster_namespace)

    if not is_running():
        raise RuntimeError("Ray is not running, please start it by `ray start --head`.")

    try:
        if cfg.stages:
            from trinity.manager.state_manager import StateManager
            from trinity.trainer import get_latest_hf_checkpoint_path

            state_manager = StateManager(path=cfg.get_checkpoint_job_dir())
            latest_stage = state_manager.load_stage().get("latest_stage", 0)
            prev_stage_checkpoint = None
            for i, stage_config in enumerate(cfg):
                if i < latest_stage:
                    logger.info(
                        "===========================================================\n"
                        f"> Skipping completed stage {i + 1}/{len(cfg.stages)}...\n"
                        "==========================================================="
                    )
                    stage_config.check_and_update()
                else:
                    logger.info(
                        "===========================================================\n"
                        f"> Starting stage {i + 1}/{len(cfg.stages)}...\n"
                        "==========================================================="
                    )
                    state_manager.save_stage(i)
                    if prev_stage_checkpoint is not None:
                        stage_config.model.model_path = prev_stage_checkpoint
                    stage_config.check_and_update()
                    run_stage(stage_config)
                    logger.info(
                        "===========================================================\n"
                        f"> Stage {i + 1}/{len(cfg.stages)} finished.\n"
                        "==========================================================="
                    )
                prev_stage_checkpoint = get_latest_hf_checkpoint_path(stage_config)
        else:
            cfg.check_and_update()
            run_stage(cfg)

    finally:
        if dlc:
            stop_ray_cluster(namespace=cluster_namespace)


app.command("debug")(debug)
app.command("studio")(studio_command)
app.add_typer(perf_app, name="perf")
app.command("view")(view_command)
app.command("convert")(convert_command)
app.command("log")(log_command)


def main() -> None:
    """The main entrypoint."""
    app()


if __name__ == "__main__":
    main()
