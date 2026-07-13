"""Per-request recording context propagation shared by model engines."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

try:
    from starlette.middleware.base import BaseHTTPMiddleware
except ModuleNotFoundError:
    BaseHTTPMiddleware = object  # type: ignore


@dataclass(frozen=True)
class RecordingContext:
    """Per-request recording metadata propagated to engine-boundary recorders."""

    record_key: Optional[str] = None
    request: Optional[dict[str, Any]] = None


recording_ctx: ContextVar[Optional[RecordingContext]] = ContextVar(
    "trinity_recording_context", default=None
)

# Set around auxiliary generate calls (logprobs recomputation, message
# conversion) so recorders skip them.
skip_recording_ctx: ContextVar[bool] = ContextVar("trinity_recording_skip", default=False)

AUTHORIZATION_HEADER = "authorization"


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an Authorization header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    if token == "EMPTY":
        return None
    return token or None


def get_recording_record_key(request: Any) -> Optional[str]:
    """Return the recording identity for an HTTP request."""
    return extract_bearer_token(request.headers.get(AUTHORIZATION_HEADER))


def get_recording_record_key_from_context() -> Optional[str]:
    """Return the current in-flight recording identity, if any."""
    ctx = recording_ctx.get()
    return None if ctx is None else ctx.record_key


def get_recording_request_from_context() -> Optional[dict[str, Any]]:
    """Return selected raw OpenAI request fields captured for recording."""
    ctx = recording_ctx.get()
    return None if ctx is None else ctx.request


class RecordingIdentityMiddleware(BaseHTTPMiddleware):
    """Capture request identity and selected raw request fields."""

    async def _get_recording_request(self, request: Any, record_key: Optional[str]):
        if record_key is None:
            return None
        try:
            body = await request.json()
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        recording_request = {}
        for field in ("messages", "tools"):
            value = body.get(field)
            if value is not None:
                recording_request[field] = value
        return recording_request or None

    async def dispatch(self, request: Any, call_next):
        record_key = get_recording_record_key(request)
        request_info = await self._get_recording_request(request, record_key)
        token = recording_ctx.set(RecordingContext(record_key=record_key, request=request_info))
        try:
            return await call_next(request)
        finally:
            recording_ctx.reset(token)
