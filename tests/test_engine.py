import asyncio
import json
import tempfile
import threading
import unittest

from chuanda_memory.engine import MemoryEngine
from chuanda_memory.extraction import parse_candidates
from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.store = MemoryStore(self.temp.name)

    async def test_background_updates_without_chat_output(self):
        called = []
        async def model(system, prompt, provider):
            called.append(json.loads(prompt))
            self.assertEqual(provider, "configured-provider")
            return json.dumps({"memories": [{"kind": "preference", "subject_id": "1001", "key": "answer_length",
                 "content": "喜欢短回答", "evidence_ids": ["m1"], "confidence": 0.95}]})
        engine = MemoryEngine(self.store, model, delay=0)
        await engine.observe(Observation("qq", "g1", "m1", "1001", "小王", "回答短一点就好", "configured-provider"))
        engine.start()
        for _ in range(100):
            if await asyncio.to_thread(lambda: bool(self.store.profile("qq", "g1", "1001")["memories"])):
                break
            await asyncio.sleep(0.01)
        await engine.close()
        self.assertEqual(len(called), 1)
        self.assertEqual(len(self.store.profile("qq", "g1", "1001")["memories"]), 1)

    async def test_failure_is_durable_and_not_busy_retried(self):
        async def fail(*args):
            raise RuntimeError("unavailable")
        engine = MemoryEngine(self.store, fail, delay=0)
        await engine.observe(Observation("qq", "g1", "m1", "1001", "小王", "你好"))
        self.assertTrue(await engine.run_once())
        self.assertFalse(await engine.run_once())
        with self.store.connection() as db:
            row = db.execute("SELECT done,attempts FROM messages").fetchone()
        self.assertEqual(tuple(row), (0, 1))

    async def test_cancelled_job_is_pending_after_restart(self):
        entered = asyncio.Event()
        async def model(*args):
            entered.set()
            await asyncio.Event().wait()
        engine = MemoryEngine(self.store, model, delay=0)
        await engine.observe(Observation("qq", "g1", "m1", "1001", "小王", "你好"))
        engine.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        await engine.close()
        reopened = MemoryStore(self.temp.name)
        self.assertIsNotNone(reopened.next_batch(delay=0, limit=16))

    async def test_shutdown_waits_for_inflight_database_thread(self):
        entered, release = threading.Event(), threading.Event()
        original = self.store.next_batch
        def blocked(**kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise TimeoutError("Test did not release the database operation")
            return original(**kwargs)
        self.store.next_batch = blocked
        async def model(*args): return '{"memories":[]}'
        engine = MemoryEngine(self.store, model, delay=0)
        engine.start()
        await asyncio.to_thread(entered.wait, 2)
        close = asyncio.create_task(engine.close())
        await asyncio.sleep(0.02)
        self.assertFalse(close.done())
        release.set()
        await asyncio.wait_for(close, timeout=2)

    def test_parser_requires_explicit_json_schema(self):
        self.assertEqual(parse_candidates('```json\n{"memories": []}\n```'), [])
        for bad in ['[]', '{}', '{"memories": "bad"}', '{"memories":[' + ','.join('{}' for _ in range(13)) + ']}']:
            with self.assertRaises(ValueError):
                parse_candidates(bad)

    async def test_database_failure_during_shutdown_does_not_restart_worker(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(**kwargs):
            entered.set()
            release.wait(timeout=3)
            raise OSError("Database operation failed while shutting down")
        self.store.next_batch = blocked
        async def model(*args): return '{"memories":[]}'
        engine = MemoryEngine(self.store, model)
        engine.start()
        await asyncio.to_thread(entered.wait, 2)
        close = asyncio.create_task(engine.close())
        await asyncio.sleep(0.02)
        release.set()
        await asyncio.wait_for(close, timeout=2)

    async def test_one_background_call_updates_target_and_shared_memory_with_custom_policy(self):
        self.store.prompt_path.write_text("优先记住有证据的交流偏好", encoding="utf-8")
        for i in range(20):
            self.store.observe(Observation("qq", "g1", "a" + str(i), "A", "成员A", "我喜欢简短的技术讨论"))
        self.store.observe(Observation("qq", "g1", "b", "B", "成员B", "我也喜欢简短的技术讨论"))
        calls = []
        async def model(system, prompt, provider):
            data = json.loads(prompt)
            calls.append(data)
            self.assertIn("优先记住有证据的交流偏好", system)
            self.assertEqual(data["target_id"], "A")
            self.assertEqual({m["sender_id"] for m in data["messages"]}, {"A"})
            self.assertIn("B", {m["sender_id"] for m in data["recent_context"]})
            return json.dumps({"memories": [
                {"kind": "preference", "subject_id": "A", "key": "answer_length", "content": "喜欢简短的技术讨论",
                 "evidence_ids": ["a0"], "confidence": 0.95},
                {"kind": "group_profile", "subject_id": "@group", "key": "discussion", "content": "近期两位成员偏好简短的技术交流",
                 "evidence_ids": ["a0", "b"], "confidence": 0.9}]})
        engine = MemoryEngine(self.store, model, delay=300, min_batch=3)
        self.assertTrue(await engine.run_once())
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.store.profile("qq", "g1", "A")["memories"]), 1)
        self.assertEqual(len(self.store.profile("qq", "g1", "@group")["memories"]), 1)
        self.assertEqual(self.store.profile("qq", "g1", "B")["stats"]["pending"], 1)
