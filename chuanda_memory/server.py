"""Optional independent service: python -m chuanda_memory serve --data-dir /data/memory."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from urllib.parse import urlparse

from .engine import LocalBackend, MemoryEngine
from .models import Observation, identifier
from .store import MemoryStore


def make_app(data_dir, token, model_call, *, delay=300, batch_size=20, min_batch=3, max_wait=1800):
    from aiohttp import web

    if not token or len(token) < 24:
        raise ValueError("Set CHUANDA_MEMORY_TOKEN to at least 24 random characters")

    @web.middleware
    async def auth(request, handler):
        if request.path != "/health" and not hmac.compare_digest(
            request.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()
        ):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            return await handler(request)
        except PermissionError:
            return web.json_response({"error": "forbidden"}, status=403)
        except (ValueError, TypeError, KeyError):
            return web.json_response({"error": "invalid_request"}, status=400)

    app = web.Application(middlewares=[auth], client_max_size=65536)
    backend = LocalBackend(MemoryEngine(MemoryStore(data_dir, readable_background=True), model_call, delay=delay,
                                        batch_size=batch_size, min_batch=min_batch, max_wait=max_wait))
    app[web.AppKey("backend", LocalBackend)] = backend

    async def lifecycle(app):
        await asyncio.to_thread(backend.store.prune)
        backend.engine.start()
        yield
        await backend.close()
    app.cleanup_ctx.append(lifecycle)

    async def health(request):
        return web.json_response({"status": "ok", "api_version": 1})

    async def dispatch(request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
        action = request.match_info["action"]
        if action == "observe":
            return web.json_response({"accepted": await backend.observe(Observation(**payload))})
        ns, scope, sender = [identifier(payload.get(k), k) for k in ("namespace", "scope", "sender_id")]
        if action == "context":
            query = payload.get("query", "")
            if not isinstance(query, str) or len(query) > 10000:
                raise ValueError("Invalid query")
            result = await backend.context(ns, scope, sender, query)
        elif action == "profile":
            result = await backend.profile(ns, scope, sender)
        elif action == "forget":
            if not isinstance(payload.get("admin", False), bool):
                raise ValueError("Invalid admin flag")
            result = {"deleted": await backend.forget(ns, scope, sender, identifier(payload.get("memory_id"), "memory_id"), admin=payload.get("admin", False))}
        elif action == "pause":
            if not isinstance(payload.get("erase", False), bool):
                raise ValueError("Invalid erase flag")
            await backend.pause(ns, scope, sender, erase=payload.get("erase", False))
            result = {"ok": True}
        elif action == "pin":
            if not isinstance(payload.get("admin", False), bool) or not isinstance(payload.get("permanent", True), bool):
                raise ValueError("Invalid pin flags")
            result = {"updated": await backend.pin(ns, scope, sender,
                      identifier(payload.get("memory_id"), "memory_id"),
                      permanent=payload.get("permanent", True), admin=payload.get("admin", False))}
        elif action == "resume":
            await backend.resume(ns, scope, sender)
            result = {"ok": True}
        elif action == "retry":
            result = {"requeued": await backend.retry_pending(ns, scope, sender)}
        else:
            raise web.HTTPNotFound()
        return web.json_response(result)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/{action}", dispatch)
    return app


class OpenAIModel:
    def __init__(self):
        self.base_url = os.environ.get("CHUANDA_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.key = os.environ.get("CHUANDA_MODEL_API_KEY", "")
        self.model = os.environ.get("CHUANDA_MODEL", "")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid CHUANDA_MODEL_BASE_URL")
        if not self.key or not self.model:
            raise ValueError("Set CHUANDA_MODEL_API_KEY and CHUANDA_MODEL for the independent service")

    async def __call__(self, system, prompt, provider_id):
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
            async with session.post(self.base_url + "/chat/completions", headers={"Authorization": "Bearer " + self.key},
                                    json={"model": self.model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 2500},
                                    allow_redirects=False) as response:
                if response.status != 200:
                    raise RuntimeError("Memory model request failed")
                raw = await response.content.read(200001)
                if len(raw) > 200000:
                    raise ValueError("Model response too large")
                return json.loads(raw)["choices"][0]["message"]["content"]


def serve(data_dir, host="127.0.0.1", port=8765, delay=300, batch_size=20, min_batch=3, max_wait=1800):
    from aiohttp import web
    app = make_app(data_dir, os.environ.get("CHUANDA_MEMORY_TOKEN", ""), OpenAIModel(), delay=delay,
                   batch_size=batch_size, min_batch=min_batch, max_wait=max_wait)
    web.run_app(app, host=host, port=port, access_log=None)
