import json
import unittest

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:
    TestClient = TestServer = None

from chuanda_memory.client import RemoteBackend
from chuanda_memory.models import Observation
from chuanda_memory.server import make_app
from tests.support import temporary_directory


@unittest.skipUnless(TestClient, "Install requirements.txt to exercise the real HTTP service")
class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.token = "test-token-" + "x" * 30
        async def model(*args): return '{"memories":[]}'
        self.app = make_app(self.temp.name, self.token, model, delay=1000)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer " + self.token}

    async def test_auth_required_and_health_contains_no_data(self):
        response = await self.client.get("/health")
        self.assertEqual(await response.json(), {"status": "ok", "api_version": 1})
        response = await self.client.post("/v1/profile", json={})
        self.assertEqual(response.status, 401)

    async def test_real_remote_client_roundtrip(self):
        backend = RemoteBackend(str(self.client.make_url("/")), self.token)
        self.assertTrue(await backend.observe(Observation("qq", "g1", "m1", "1001", "小王", "你好")))
        self.assertFalse(await backend.observe(Observation("qq", "g1", "m1", "1001", "小王", "你好")))
        profile = await backend.profile("qq", "g1", "1001")
        self.assertEqual(profile["name"], "小王")
        context = await backend.context("qq", "g2", "1001", "你好")
        self.assertFalse(context["memories"])
        await backend.pause("qq", "g1", "1001", erase=True)
        self.assertTrue((await backend.profile("qq", "g1", "1001"))["paused"])
        await backend.resume("qq", "g1", "1001")
        self.assertFalse((await backend.profile("qq", "g1", "1001"))["paused"])

    async def test_invalid_input_is_rejected(self):
        for payload in [{}, [], {"namespace": "qq", "scope": "g1", "sender_id": "1001", "query": []}]:
            response = await self.client.post("/v1/context", headers=self.headers, json=payload)
            self.assertEqual(response.status, 400)
        response = await self.client.post("/v1/observe", headers=self.headers, json={
            "namespace": "qq", "scope": "g1", "message_id": "m1", "sender_id": "@group", "sender_name": "x", "text": "fake"})
        self.assertEqual(response.status, 400)

    async def test_unknown_action_and_payload_limit(self):
        response = await self.client.post("/v1/missing", headers=self.headers,
                                          json={"namespace": "qq", "scope": "g1", "sender_id": "1001"})
        self.assertEqual(response.status, 404)
        response = await self.client.post("/v1/context", headers=self.headers, data="x" * 70000)
        self.assertEqual(response.status, 413)

    def test_service_rejects_short_or_missing_token(self):
        for token in ["", "short"]:
            with self.assertRaises(ValueError):
                make_app(self.temp.name, token, None)
