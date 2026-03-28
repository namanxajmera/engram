import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._api_key = os.environ.get("MEMORY_API_KEY")
        if not self._api_key:
            logger.warning("MEMORY_API_KEY not set — all non-health requests will be rejected")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/ui") or request.method == "OPTIONS":
            return await call_next(request)

        if not self._api_key:
            return JSONResponse({"error": "MEMORY_API_KEY not configured"}, status_code=503)

        auth_header = request.headers.get("authorization", "")
        token = auth_header.removeprefix("Bearer ")

        if not token or not hmac.compare_digest(token, self._api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)
