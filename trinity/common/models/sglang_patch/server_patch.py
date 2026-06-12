from __future__ import annotations

import asyncio
import os
import time
from logging import Logger
from typing import Any, Callable, Coroutine, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Response
from fastapi.dependencies.utils import (
    _should_embed_body_fields,
    get_body_field,
    get_dependant,
    get_flat_dependant,
)
from fastapi.routing import APIRoute, request_response
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

from trinity.common.models.sglang_patch.openai_api_patch import (
    ChatCompletionRequest as PatchedChatCompletionRequest,
)
from trinity.common.models.sglang_patch.openai_api_patch import (
    ChatCompletionResponse as PatchedChatCompletionResponse,
)
from trinity.common.models.sglang_patch.openai_api_patch import (
    ChatCompletionResponseChoice as PatchedChatCompletionResponseChoice,
)
from trinity.common.models.sglang_patch.openai_api_patch import PatchedOpenAIServingChat
from trinity.utils.distributed import get_endpoint


def _refresh_chat_completion_routes() -> None:
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path not in {"/v1/chat/completions", "/invocations"}:
            continue

        route.endpoint.__annotations__["request"] = PatchedChatCompletionRequest
        route.dependant = get_dependant(
            path=route.path_format,
            call=route.endpoint,
            scope="function",
        )
        flat_dependant = get_flat_dependant(route.dependant)
        embed_body_fields = _should_embed_body_fields(flat_dependant.body_params)
        setattr(route, "_flat_dependant", flat_dependant)
        setattr(route, "_embed_body_fields", embed_body_fields)
        route.body_field = get_body_field(
            flat_dependant=flat_dependant,
            name=route.unique_id,
            embed_body_fields=embed_body_fields,
        )
        route.app = request_response(route.get_route_handler())


def _apply_openai_api_monkey_patch() -> None:
    import sglang.srt.entrypoints.http_server as http_server_module
    import sglang.srt.entrypoints.openai.protocol as protocol_module
    import sglang.srt.entrypoints.openai.serving_chat as serving_chat_module

    protocol_module.ChatCompletionRequest = PatchedChatCompletionRequest
    serving_chat_module.ChatCompletionRequest = PatchedChatCompletionRequest
    serving_chat_module.ChatCompletionResponse = PatchedChatCompletionResponse
    serving_chat_module.ChatCompletionResponseChoice = PatchedChatCompletionResponseChoice
    http_server_module.ChatCompletionRequest = PatchedChatCompletionRequest
    http_server_module.OpenAIServingChat = PatchedOpenAIServingChat

    _refresh_chat_completion_routes()


def _create_cleanup(
    server_args: ServerArgs,
    tokenizer_manager,
    subprocess_watchdog: Optional[SubprocessWatchdog],
    logger: Logger,
) -> Callable[[], Coroutine[None, None, None]]:
    cleaned_up = False

    async def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True

        if subprocess_watchdog is not None:
            subprocess_watchdog.stop()
            processes = getattr(subprocess_watchdog, "_processes", [])

            # Send SIGTERM to all alive processes simultaneously
            alive_processes = [
                p for p in processes if p is not None and p.pid is not None and p.is_alive()
            ]

            for process in alive_processes:
                try:
                    process.terminate()
                except Exception as exc:
                    logger.warning("Failed to send SIGTERM: %s", exc)

            if alive_processes:
                # Wait for graceful exit
                graceful_timeout = 5
                start_time = time.time()
                while time.time() - start_time < graceful_timeout:
                    if not any(p.is_alive() for p in alive_processes):
                        logger.info("All SGLang subprocesses exited gracefully.")
                        break
                    await asyncio.sleep(0.2)

                # Force kill remaining processes
                still_alive = [p for p in alive_processes if p.is_alive()]
                for process in still_alive:
                    try:
                        kill_process_tree(process.pid, wait_timeout=60)
                    except Exception as exc:
                        logger.warning(
                            "Failed to terminate SGLang child process %s: %s",
                            process.pid,
                            exc,
                        )
        elif server_args.node_rank > 0:
            kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)

        if tokenizer_manager is not None and hasattr(tokenizer_manager, "_subprocess_watchdog"):
            tokenizer_manager._subprocess_watchdog = None

        import sglang.srt.entrypoints.http_server as http_server_module

        http_server_module._global_state = None

    return cleanup


def _create_server_task(
    server: uvicorn.Server,
    cleanup: Callable[[], Coroutine[None, None, None]],
    logger: Logger,
) -> "asyncio.Task[None]":
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


def _run_dummy_health_server(
    server_args: ServerArgs,
    subprocess_watchdog: Optional[SubprocessWatchdog],
    logger: Logger,
) -> "asyncio.Task[None]":
    dummy_app = FastAPI()

    @dummy_app.get("/ping")
    async def ping() -> Response:
        return Response(status_code=200)

    @dummy_app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @dummy_app.get("/health_generate")
    async def health_generate() -> Response:
        return Response(status_code=200)

    if server_args.enable_metrics:
        from sglang.srt.utils.common import add_prometheus_middleware, enable_func_timer

        add_prometheus_middleware(dummy_app)
        enable_func_timer()

    set_uvicorn_logging_configs(server_args)
    server = uvicorn.Server(
        uvicorn.Config(
            dummy_app,
            host=server_args.host,
            port=server_args.port,
            timeout_keep_alive=5,
            loop="auto",
            log_level=server_args.log_level_http or server_args.log_level,
        )
    )
    logger.info(
        "Starting SGLang worker health server on %s:%s for node_rank=%s.",
        server_args.host,
        server_args.port,
        server_args.node_rank,
    )
    return _create_server_task(
        server,
        _create_cleanup(server_args, None, subprocess_watchdog, logger),
        logger,
    )


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
    return _create_server_task(
        server,
        _create_cleanup(server_args, tokenizer_manager, subprocess_watchdog, logger),
        logger,
    )


def get_api_server(
    host: str,
    port: int,
    model_path: Optional[str],
    tensor_parallel_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    extra_engine_args: Optional[Dict[str, Any]],
    dtype: str,
    served_model_name: Optional[str],
    mem_fraction_static: float,
    trust_remote_code: bool,
    context_length: Optional[int],
    enable_multimodal: bool,
    enable_return_routed_experts: bool,
    api_key: str,
    nnodes: int,
    node_rank: int,
    master_addr: Optional[str],
    master_port: Optional[int],
    logger: Logger,
) -> "asyncio.Task[None]":
    _apply_openai_api_monkey_patch()

    if model_path is None:
        raise ValueError("model_path must be provided when launching the SGLang API server.")

    server_args_kwargs: Dict[str, Any] = dict(
        host=host,
        port=port,
        model_path=model_path,
        tp_size=tensor_parallel_size,
        dp_size=data_parallel_size,
        pp_size=pipeline_parallel_size,
        ep_size=tensor_parallel_size if enable_expert_parallel else 1,
        dtype=dtype,
        served_model_name=served_model_name,
        mem_fraction_static=mem_fraction_static,
        trust_remote_code=trust_remote_code,
        context_length=context_length,
        enable_multimodal=enable_multimodal,
        enable_return_routed_experts=enable_return_routed_experts,
        skip_server_warmup=True,
        disable_piecewise_cuda_graph=True,
        api_key=api_key,
        nnodes=nnodes,
        node_rank=node_rank,
        dist_init_addr=(
            get_endpoint(master_addr, master_port) if master_addr and master_port else None
        ),
        # Aliyun DSW / DLC requires this setting to avoid NCCL error
        enable_symm_mem=True,
        device="cuda",
    )
    if extra_engine_args:
        server_args_kwargs.update(extra_engine_args)
    server_args = ServerArgs(**server_args_kwargs)

    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

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

    if server_args.node_rank > 0:
        return _run_dummy_health_server(
            server_args=server_args,
            subprocess_watchdog=subprocess_watchdog,
            logger=logger,
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
