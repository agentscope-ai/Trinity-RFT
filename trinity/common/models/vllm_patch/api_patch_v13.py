"""Patch for vllm OpenAI API server. Only for vllm versions >= 0.13.0.
"""
import asyncio
import functools
import logging
from typing import Optional

import vllm
import vllm.envs as envs
from packaging.version import parse as parse_version
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    create_server_unix_socket,
    init_app_state,
    validate_api_server_args,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.utils import log_non_default_args
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

from trinity.common.models.vllm_patch import get_vllm_version


def setup_server_in_ray(args, logger):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""

    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock


def dummy_add_signal_handler(self, *args, **kwargs):
    # DO NOTHING HERE
    pass


async def run_server_worker_in_ray(
    listen_address,
    sock,
    args,
    engine_client,
    logger,
) -> None:
    # Modified from vllm.entrypoints.openai.api_server.run_server_worker
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    app = build_app(args)

    await init_app_state(engine_client, app.state, args)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler = functools.partial(dummy_add_signal_handler, loop)

    logger.info(
        "Starting vLLM API server %d on %s",
        engine_client.vllm_config.parallel_config._api_process_rank,
        listen_address,
    )

    shutdown_task = await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
    )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


async def run_server_in_ray(args, engine_client, logger):
    # Modified from vllm.entrypoints.openai.api_server.run_server
    listen_address, sock = setup_server_in_ray(args, logger)
    logger.info("vLLM API server listening on %s", listen_address)
    await run_server_worker_in_ray(listen_address, sock, args, engine_client, logger)


async def run_api_server_in_ray_actor_v13(
    async_llm,
    host: str,
    port: int,
    model_path: str,
    logger: logging.Logger,
    chat_template: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    enable_log_requests: bool = False,
):
    vllm_version = get_vllm_version()
    if vllm_version < parse_version("0.13.0"):
        raise ValueError(
            f"Unsupported vllm version: {vllm.__version__}. "
            "This patch requires vllm version >= 0.13.0"
        )

    parser = FlexibleArgumentParser(description="Run the OpenAI API server.")
    args = make_arg_parser(parser)
    cli_args = [
        "--host",
        str(host),
        "--port",
        str(port),
        "--model",
        model_path,
        "--enable-server-load-tracking",  # enable tracking for load balancing
    ]
    if enable_log_requests:
        cli_args.append("--enable-log-requests")
    if enable_auto_tool_choice:
        cli_args.append("--enable-auto-tool-choice")
    if tool_call_parser:
        cli_args.extend(["--tool-call-parser", tool_call_parser])
    if reasoning_parser:
        cli_args.extend(["--reasoning-parser", reasoning_parser])
    if chat_template:
        cli_args.extend(["--chat-template", chat_template])
    args = parser.parse_args(cli_args)
    args.structured_outputs_config.reasoning_parser = reasoning_parser
    if chat_template:
        args.chat_template = chat_template
    logger.info(f"Starting vLLM OpenAI API server with args: {args}")
    await run_server_in_ray(args, async_llm, logger)
