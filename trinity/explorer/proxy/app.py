import traceback
from contextlib import asynccontextmanager
from typing import Dict

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
    app.state.models_cache = None
    yield
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan)


def _build_forward_headers(request: Request) -> Dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _build_downstream_headers(headers: httpx.Headers) -> Dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def _build_json_or_text_response(upstream_response: httpx.Response):
    headers = _build_downstream_headers(upstream_response.headers)
    content_type = upstream_response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return JSONResponse(
            status_code=upstream_response.status_code,
            content=upstream_response.json(),
            headers=headers,
        )
    return Response(
        status_code=upstream_response.status_code,
        content=upstream_response.content,
        headers=headers,
        media_type=upstream_response.headers.get("content-type"),
    )


async def _proxy_chat_stream(
    request: Request,
    upstream_response: httpx.Response,
):
    """Pure passthrough: stream the upstream SSE bytes to the client unchanged.

    Experience capture is handled in-process by the vLLM recorder (wrapping
    ``engine_client.generate``), so the proxy no longer parses/aggregates the
    stream here.
    """

    async def iterator():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        content=iterator(),
        status_code=upstream_response.status_code,
        headers=_build_downstream_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


# Forward OpenAI requests to a model instance
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    forward_headers = _build_forward_headers(request)
    # Temperature is a policy knob controlled by the explorer config; override
    # the client's value. (Experience capture — token_ids/logprobs — is handled
    # in-process by the vLLM recorder, so we no longer force them onto the wire.)
    request_data["temperature"] = request.app.state.temperature

    url, _ = await request.app.state.service.allocate_model()

    if request_data.get("stream", False):
        # Streaming: passthrough the upstream SSE bytes unchanged.
        try:
            upstream_request = request.app.state.http_client.build_request(
                method="POST",
                url=f"{url}/v1/chat/completions",
                json=request_data,
                headers=forward_headers,
                timeout=request.app.state.inference_timeout,
            )
            upstream_response = await request.app.state.http_client.send(
                upstream_request,
                stream=True,
            )
        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": f"Upstream timeout when forwarding request to model at {url}.",
                        "type": "upstream_timeout",
                        "code": "gateway_timeout",
                    }
                },
            )
        except httpx.RequestError:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Failed to connect upstream model at {url}: {traceback.format_exc()}",
                        "type": "upstream_connection_error",
                        "code": "bad_gateway",
                    }
                },
            )

        return await _proxy_chat_stream(
            request=request,
            upstream_response=upstream_response,
        )

    try:
        resp = await request.app.state.http_client.post(
            f"{url}/v1/chat/completions",
            json=request_data,
            headers=forward_headers,
            timeout=request.app.state.inference_timeout,
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": f"Upstream timeout when forwarding request to model at {url}.",
                    "type": "upstream_timeout",
                    "code": "gateway_timeout",
                }
            },
        )
    except httpx.RequestError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Failed to connect upstream model at {url}: {traceback.format_exc()}",
                    "type": "upstream_connection_error",
                    "code": "bad_gateway",
                }
            },
        )

    try:
        resp_data = resp.json()
    except ValueError:
        return _build_json_or_text_response(resp)

    if resp.status_code >= 400:
        return JSONResponse(
            status_code=resp.status_code,
            content=resp_data,
            headers=_build_downstream_headers(resp.headers),
        )

    # Non-streaming success: forward unchanged. Experience capture happens
    # in-process at the vLLM engine boundary (the recorder wraps
    # engine_client.generate), so nothing to record here.
    return JSONResponse(
        status_code=resp.status_code,
        content=resp_data,
        headers=_build_downstream_headers(resp.headers),
    )


@app.get("/v1/models")
async def show_available_models(request: Request):
    if request.app.state.models_cache is not None:
        return JSONResponse(content=request.app.state.models_cache)

    url, _ = await request.app.state.service.allocate_model(increase_count=False)
    try:
        resp = await request.app.state.http_client.get(f"{url}/v1/models")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch upstream models: {str(e)}")

    if resp.status_code >= 400:
        return _build_json_or_text_response(resp)

    request.app.state.models_cache = resp.json()
    return JSONResponse(content=request.app.state.models_cache)


@app.get("/health")
async def health(request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/metrics")
async def metrics(request: Request):
    """Get the metrics of the service."""
    metrics = request.app.state.service.collect_metrics()
    metrics["explore_step_num"] = request.app.state.service.explorer.explore_step_num
    return JSONResponse(content=metrics)


async def serve_http(app: FastAPI, host: str, port: int) -> None:
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def run_app(service, listen_address: str, port: int) -> None:
    app.state.service = service
    app.state.temperature = service.explorer.config.model.temperature
    app.state.inference_timeout = service.explorer.config.synchronizer.sync_timeout
    await serve_http(app, listen_address, port)
