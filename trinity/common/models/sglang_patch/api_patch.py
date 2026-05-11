from __future__ import annotations

import asyncio
from logging import Logger
from typing import Callable, Dict, List, Optional

import uvicorn
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server import (
    _execute_server_warmup,
    _GlobalState,
    add_prometheus_track_response_middleware,
    app,
    app_has_admin_force_endpoints,
    envs,
    set_global_state,
    set_uvicorn_logging_configs,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.watchdog import SubprocessWatchdog


def _setup_and_run_http_server(  # noqa: C901
    server_args: ServerArgs,
    tokenizer_manager,
    template_manager,
    port_args,
    scheduler_infos: List[Dict],
    subprocess_watchdog: Optional[SubprocessWatchdog],
    logger: Logger,
    server: uvicorn.Server,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
) -> "asyncio.Task[None]":
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )
    del port_args

    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    setattr(app, "is_single_tokenizer_mode", True)
    setattr(app, "server_args", server_args)
    setattr(
        app,
        "warmup_thread_kwargs",
        {
            "server_args": server_args,
            "launch_callback": launch_callback,
            "execute_warmup_func": execute_warmup_func,
        },
    )

    if server_args.api_key or server_args.admin_api_key or app_has_admin_force_endpoints(app):
        from sglang.srt.utils.auth import add_api_key_middleware

        add_api_key_middleware(
            app,
            api_key=server_args.api_key,
            admin_api_key=server_args.admin_api_key,
        )

    set_uvicorn_logging_configs(server_args)
    if server_args.ssl_certfile:
        logger.info(
            "SSL enabled: certfile=%s, keyfile=%s",
            server_args.ssl_certfile,
            server_args.ssl_keyfile,
        )

    cleaned_up = False

    async def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True

        if subprocess_watchdog is not None:
            subprocess_watchdog.stop()
            for process in getattr(subprocess_watchdog, "_processes", []):
                if process is None or process.pid is None:
                    continue
                try:
                    kill_process_tree(process.pid, wait_timeout=60)
                except Exception as exc:
                    logger.warning(
                        "Failed to terminate SGLang child process %s: %s",
                        process.pid,
                        exc,
                    )
        if tokenizer_manager is not None and hasattr(tokenizer_manager, "_subprocess_watchdog"):
            tokenizer_manager._subprocess_watchdog = None

        import sglang.srt.entrypoints.http_server as http_server_module

        http_server_module._global_state = None

    task = asyncio.create_task(server.serve())

    def _cleanup_task(task: "asyncio.Task[None]") -> None:
        if task.cancelled():
            asyncio.create_task(cleanup())
            return
        if task.exception() is not None:
            logger.warning("Embedded SGLang HTTP server exited with error: %s", task.exception())
        asyncio.create_task(cleanup())

    task.add_done_callback(_cleanup_task)
    return task


def get_api_server(
    server_args: ServerArgs,
    logger: Logger,
) -> "asyncio.Task[None]":
    if server_args.enable_http2:
        raise NotImplementedError("Embedded SGLang server does not support HTTP/2 yet.")
    if server_args.tokenizer_worker_num != 1:
        raise NotImplementedError(
            "Embedded SGLang server currently supports tokenizer_worker_num == 1 only."
        )
    if server_args.enable_ssl_refresh:
        raise NotImplementedError("Embedded SGLang server does not support SSL refresh yet.")

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=Engine.init_tokenizer_manager_func,
        run_scheduler_process_func=Engine.run_scheduler_process_func,
        run_detokenizer_process_func=Engine.run_detokenizer_process_func,
    )

    config = uvicorn.Config(
        app,
        host=server_args.host,
        port=server_args.port,
        root_path=server_args.fastapi_root_path,
        log_level=server_args.log_level_http or server_args.log_level,
        timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
        loop="uvloop",
        ssl_keyfile=server_args.ssl_keyfile,
        ssl_certfile=server_args.ssl_certfile,
        ssl_ca_certs=server_args.ssl_ca_certs,
        ssl_keyfile_password=server_args.ssl_keyfile_password,
    )
    server = uvicorn.Server(config)
    return _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        logger,
        server,
    )
