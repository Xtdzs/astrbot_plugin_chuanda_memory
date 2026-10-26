from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .extraction import SYSTEM_PROMPT, extraction_prompt, parse_candidates
from .models import Observation
from .store import MemoryStore

ModelCall = Callable[[str, str, str], Awaitable[str]]
logger = logging.getLogger(__name__)


async def settled_thread_call(function, *args, **kwargs):
    """Cancellation must not leave an SQLite operation running after shutdown."""
    def execute():
        try:
            return True, function(*args, **kwargs)
        except Exception as error:
            return False, error
    work = asyncio.create_task(asyncio.to_thread(execute))
    try:
        succeeded, result = await asyncio.shield(work)
    except asyncio.CancelledError as cancellation:
        await work  # Wait for commit/rollback; preserve cancellation even on DB failure.
        raise cancellation
    if not succeeded:
        raise result
    return result


class MemoryEngine:
    def __init__(self, store: MemoryStore, model_call: ModelCall, *, delay=20, batch_size=16, timeout=45, retention_days=14):
        self.store = store
        self.model_call = model_call
        self.delay = max(0, float(delay))
        self.batch_size = max(1, min(40, int(batch_size)))
        self.timeout = max(1, float(timeout))
        self.retention_days = max(1, int(retention_days))
        self._task = None
        self._wake = asyncio.Event()
        self._run_lock = asyncio.Lock()

    async def observe(self, message: Observation):
        added = await settled_thread_call(self.store.observe, message)
        if added:
            self._wake.set()
        return added

    async def run_once(self):
        # A single worker per database; the lock also prevents manual flush races.
        async with self._run_lock:
            batch = await settled_thread_call(self.store.next_batch, delay=self.delay, limit=self.batch_size)
            if not batch:
                return False
            provider = next((m["provider_id"] for m in reversed(batch["messages"]) if m["provider_id"]), "")
            try:
                result = await asyncio.wait_for(
                    self.model_call(SYSTEM_PROMPT, extraction_prompt(batch["messages"], batch["existing"]), provider),
                    timeout=self.timeout,
                )
                candidates = parse_candidates(result)
                written, rejected = await settled_thread_call(self.store.apply_batch, batch, candidates)
                logger.debug("Chuanda memory batch: written=%s rejected=%s", written, rejected)
            except asyncio.CancelledError:
                raise  # Observations remain pending across a reload or restart.
            except Exception as exc:
                await settled_thread_call(self.store.fail_batch, batch)
                # Provider exceptions can contain API keys or message text; don't log their body.
                logger.warning("Chuanda memory extraction deferred (%s)", type(exc).__name__)
            return True

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker(), name="chuanda-memory-worker")

    async def _worker(self):
        count = 0
        while True:
            try:
                had_work = await self.run_once()
                if had_work:
                    count += 1
                    if count % 20 == 0:
                        await settled_thread_call(self.store.prune, self.retention_days)
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Chuanda memory worker retry (%s)", type(exc).__name__)
                await asyncio.sleep(1)

    async def close(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class LocalBackend:
    def __init__(self, engine: MemoryEngine, *, recall_limit=8, char_budget=5000, cooldown=21600):
        self.engine = engine
        self.store = engine.store
        self.recall_limit = recall_limit
        self.char_budget = char_budget
        self.cooldown = cooldown

    async def observe(self, observation):
        return await self.engine.observe(observation)

    async def context(self, namespace, scope, sender_id, query):
        return await settled_thread_call(self.store.context, namespace, scope, sender_id, query,
                                       limit=self.recall_limit, char_budget=self.char_budget, cooldown=self.cooldown)

    async def profile(self, namespace, scope, sender_id):
        return await settled_thread_call(self.store.profile, namespace, scope, sender_id)

    async def forget(self, namespace, scope, sender_id, memory_id, *, admin=False):
        return await settled_thread_call(self.store.forget, namespace, scope, sender_id, memory_id, admin=admin)

    async def pause(self, namespace, scope, sender_id, *, erase=False):
        return await settled_thread_call(self.store.pause, namespace, scope, sender_id, erase=erase)

    async def resume(self, namespace, scope, sender_id):
        return await settled_thread_call(self.store.resume, namespace, scope, sender_id)

    async def close(self):
        await self.engine.close()
