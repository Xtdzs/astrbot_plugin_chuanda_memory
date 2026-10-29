"""Private adapter-to-service client. Its token must never be exposed to the LLM."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None  # Never forward the private service bearer token to another URL.


class RemoteBackend:
    def __init__(self, base_url, token, *, timeout=3):
        url = urlparse(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("service_url must be an HTTP(S) URL without credentials or query")
        if not token or len(token) < 24:
            raise ValueError("Memory service token must contain at least 24 characters")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, action, payload):
        request = Request(self.base_url + "/v1/" + action,
                          data=json.dumps(payload, ensure_ascii=False).encode(),
                          headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
                          method="POST")
        with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
            body = response.read(1_000_001)
            if len(body) > 1_000_000:
                raise ValueError("Memory service response too large")
            return json.loads(body)

    async def call(self, action, payload):
        return await asyncio.to_thread(self._post, action, payload)

    async def observe(self, observation):
        return (await self.call("observe", observation.to_dict()))["accepted"]

    async def context(self, namespace, scope, sender_id, query):
        return await self.call("context", dict(namespace=namespace, scope=scope, sender_id=sender_id, query=query))

    async def profile(self, namespace, scope, sender_id):
        return await self.call("profile", dict(namespace=namespace, scope=scope, sender_id=sender_id))

    async def forget(self, namespace, scope, sender_id, memory_id, *, admin=False):
        return (await self.call("forget", dict(namespace=namespace, scope=scope, sender_id=sender_id,
                                               memory_id=memory_id, admin=admin)))["deleted"]

    async def pause(self, namespace, scope, sender_id, *, erase=False):
        await self.call("pause", dict(namespace=namespace, scope=scope, sender_id=sender_id, erase=erase))

    async def pin(self, namespace, scope, sender_id, memory_id, *, permanent=True, admin=False):
        return (await self.call("pin", dict(namespace=namespace, scope=scope, sender_id=sender_id,
                                            memory_id=memory_id, permanent=permanent, admin=admin)))["updated"]

    async def resume(self, namespace, scope, sender_id):
        await self.call("resume", dict(namespace=namespace, scope=scope, sender_id=sender_id))

    async def retry_pending(self, namespace, scope, sender_id):
        return (await self.call("retry", dict(namespace=namespace, scope=scope, sender_id=sender_id)))["requeued"]

    async def close(self):
        pass
