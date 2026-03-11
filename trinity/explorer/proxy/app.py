import json
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

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


def _consume_sse_line(line: str, aggregate: Dict[str, Any]) -> None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return

    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return

    if isinstance(data.get("id"), str) and data["id"]:
        aggregate["id"] = data["id"]

    prompt_token_ids = data.get("prompt_token_ids")
    if isinstance(prompt_token_ids, list) and prompt_token_ids:
        aggregate["prompt_token_ids"] = prompt_token_ids

    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue

        choice_index = choice.get("index", 0)
        if not isinstance(choice_index, int):
            choice_index = 0

        choice_acc = aggregate["choices"].setdefault(
            choice_index,
            {
                "index": choice_index,
                "token_ids": [],
                "logprobs": {"content": []},
            },
        )

        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list) and token_ids:
            choice_acc["token_ids"].extend(token_ids)

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            content = logprobs.get("content")
            if isinstance(content, list) and content:
                choice_acc["logprobs"]["content"].extend(content)


def _finalize_stream_aggregate(aggregate: Dict[str, Any]) -> Dict[str, Any] | None:
    prompt_token_ids = aggregate.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        return None

    ordered_choices = []
    for _, choice in sorted(aggregate["choices"].items(), key=lambda item: item[0]):
        if not choice.get("token_ids"):
            continue
        ordered_choices.append(choice)

    if not ordered_choices:
        return None

    return {
        "id": aggregate.get("id", ""),
        "prompt_token_ids": prompt_token_ids,
        "choices": ordered_choices,
    }


async def _proxy_chat_stream_with_experience(
    request: Request,
    upstream_response: httpx.Response,
    model_version: int,
):
    async def iterator():
        stream_buffer = ""
        aggregate = {
            "id": "",
            "prompt_token_ids": [],
            "choices": {},
        }

        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    stream_buffer += chunk.decode("utf-8", errors="ignore")
                    while "\n" in stream_buffer:
                        line, stream_buffer = stream_buffer.split("\n", 1)
                        _consume_sse_line(line.rstrip("\r"), aggregate)
                    yield chunk
        finally:
            if stream_buffer:
                _consume_sse_line(stream_buffer.rstrip("\r"), aggregate)

            await upstream_response.aclose()

            experience_response = _finalize_stream_aggregate(aggregate)
            if experience_response is not None:
                try:
                    await request.app.state.service.record_experience(
                        experience_response,
                        model_version,
                    )
                except Exception:
                    pass

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
    # for experience data recording, we need to return token ids and logprobs
    request_data["return_token_ids"] = True
    request_data["logprobs"] = True
    # temperature must be set from config, ignore user's input
    request_data["temperature"] = request.app.state.temperature

    url, model_version = await request.app.state.service.allocate_model()

    if request_data.get("stream", False):
        # For streaming response, we need to handle it differently to aggregate experience data
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

        return await _proxy_chat_stream_with_experience(
            request=request,
            upstream_response=upstream_response,
            model_version=model_version,
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

    await request.app.state.service.record_experience(resp_data, model_version)
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


@app.post("/feedback")
async def feedback(request: Request):
    """Receive feedback for the current session."""
    body = await request.json()
    reward = body.get("reward")
    msg_ids = body.get("msg_ids")
    task_id = body.get("task_id")
    run_id = body.get("run_id", 0)
    if msg_ids is None or reward is None:
        return JSONResponse(status_code=400, content={"error": "msg_ids and reward are required"})
    if not isinstance(msg_ids, list) or not isinstance(reward, (int, float)):
        return JSONResponse(
            status_code=400, content={"error": "msg_ids must be a list and reward must be a number"}
        )
    await request.app.state.service.record_feedback(
        reward=reward, msg_ids=msg_ids, task_id=task_id, run_id=run_id
    )
    return JSONResponse(content={"status": "success"})


@app.post("/commit")
async def commit(request: Request):
    """Commit the current experiences."""
    await request.app.state.service.submit_experiences()
    return JSONResponse(content={"status": "success"})


async def serve_http(app: FastAPI, host: str, port: int) -> None:
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def run_app(service, listen_address: str, port: int) -> None:
    app.state.service = service
    app.state.temperature = service.explorer.config.model.temperature
    app.state.inference_timeout = service.explorer.config.synchronizer.sync_timeout
    await serve_http(app, listen_address, port)
