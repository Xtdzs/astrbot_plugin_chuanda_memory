"""Host boundary tests with explicit doubles; these are not a live QQ deployment."""
import asyncio
import importlib.util
import json
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from chuanda_memory.engine import LocalBackend, MemoryEngine
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


def load_plugin():
    modules = {}
    for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.provider", "astrbot.api.star",
                 "astrbot.core", "astrbot.core.agent", "astrbot.core.agent.message"):
        modules[name] = types.ModuleType(name)
    api = modules["astrbot.api"]
    api.AstrBotConfig = dict
    api.logger = logging.getLogger("test-adapter")
    def decorator(*args, **kwargs):
        return lambda function: function
    modules["astrbot.api.event"].filter = SimpleNamespace(
        event_message_type=decorator, on_llm_request=decorator, command=decorator,
        EventMessageType=SimpleNamespace(ALL="all"))
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.provider"].ProviderRequest = SimpleNamespace
    class Star:
        def __init__(self, context):
            self.context = context
    star = modules["astrbot.api.star"]
    star.Context = object
    star.Star = Star
    star.StarTools = SimpleNamespace()
    star.register = decorator
    class TextPart:
        def __init__(self, text):
            self.text = text
            self._no_save = False
        def mark_as_temp(self):
            self._no_save = True
            return self
    modules["astrbot.core.agent.message"].TextPart = TextPart
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("tested_chuanda_plugin")
    package.__path__ = [str(root)]
    modules[package.__name__] = package
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(package.__name__ + ".main", root / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self, *, sender="1001", name="小王", group="123", text="我喜欢 Rust", mid="m1", admin=False, instance="account1"):
        self.sender, self.name, self.group, self.text, self.admin, self.instance = sender, name, group, text, admin, instance
        self.message_obj = SimpleNamespace(message_id=mid)
        self.extras = {}
        self.unified_msg_origin = "conversation-can-change"
        self.sent = []
        self.stopped = False
    def get_sender_id(self): return self.sender
    def get_sender_name(self): return self.name
    def get_self_id(self): return "bot1"
    def get_group_id(self): return self.group
    def get_platform_name(self): return "aiocqhttp"
    def get_platform_id(self): return self.instance
    def get_message_str(self): return self.text
    def get_messages(self): return [SimpleNamespace(type="Plain", text=self.text)]
    def get_extra(self, key): return self.extras.get(key)
    def set_extra(self, key, value): self.extras[key] = value
    def is_admin(self): return self.admin
    def stop_event(self): self.stopped = True
    def plain_result(self, text): return text


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.module = load_plugin()
        # Model an installed code directory beside, rather than above, its durable data.
        self.module.__file__ = str(Path(self.temp.name) / "plugin_source" / "main.py")
        class Host:
            def __init__(self): self.calls = []
            async def get_current_chat_provider_id(self, **kwargs): return "provider1"
            async def llm_generate(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(completion_text='{"memories":[]}')
        self.host = Host()
        self.plugin = self.module.ChuandaMemoryPlugin(self.host, {"data_dir": self.temp.name, "batch_delay_seconds": 3600})
        await self.plugin.initialize()
        self.addAsyncCleanup(self.plugin.terminate)

    async def test_observe_and_llm_hook_are_deduplicated_and_silent(self):
        event = Event()
        await self.plugin.observe_message(event)
        request = SimpleNamespace(system_prompt="现有人格", extra_user_content_parts=[])
        await self.plugin.inject_memory(event, request)
        await self.plugin.inject_memory(event, request)
        self.assertEqual(len(request.extra_user_content_parts), 1)
        self.assertTrue(request.extra_user_content_parts[0]._no_save)
        self.assertTrue(request.system_prompt.startswith("现有人格"))
        self.assertEqual(event.sent, [])
        self.assertEqual(self.host.calls, [])
        with self.plugin.backend.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 1)

    async def test_chat_failure_does_not_block_response(self):
        class Broken:
            async def observe(self, event): raise ConnectionError("unavailable")
        await self.plugin.backend.close()
        self.plugin.backend = Broken()
        request = SimpleNamespace(system_prompt="original", extra_user_content_parts=[])
        await self.plugin.inject_memory(Event(), request)
        self.assertEqual(request.system_prompt, "original")
        self.plugin.backend = None

    async def test_scopes_ignore_conversation_reset(self):
        first = Event()
        before = self.plugin._identity(first)
        first.unified_msg_origin = "new-session"
        self.assertEqual(self.plugin._identity(first), before)
        self.assertNotEqual(self.plugin._identity(Event(group="")), before)
        self.assertNotEqual(self.plugin._identity(Event(instance="account2")), before)

    async def test_no_self_messages_or_disallowed_groups(self):
        self.plugin.config["allowed_groups"] = ["123"]
        await self.plugin.observe_message(Event(sender="bot1"))
        await self.plugin.observe_message(Event(group="other"))
        with self.plugin.backend.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    async def test_background_model_does_not_use_agent_tools_or_history(self):
        await self.plugin._call_model("extract", "raw evidence", "provider1")
        self.assertIsNone(self.host.calls[0]["tools"])
        self.assertEqual(self.host.calls[0]["contexts"], [])
        self.assertEqual(self.host.calls[0]["system_prompt"], "extract")

    async def test_command_clear_is_scoped_and_stops_chat_pipeline(self):
        event = Event()
        await self.plugin.observe_message(event)
        replies = [reply async for reply in self.plugin.memory_command(event, "清空")]
        self.assertTrue(event.stopped)
        self.assertEqual(len(replies), 1)
        ns, scope, sender = self.plugin._identity(event)
        self.assertTrue((await self.plugin.backend.profile(ns, scope, sender))["paused"])

    async def test_message_to_background_memory_to_next_request(self):
        async def model(**kwargs):
            evidence = json.loads(kwargs["prompt"])["messages"][0]
            return SimpleNamespace(completion_text=json.dumps({"memories": [{
                "kind": "preference", "subject_id": evidence["sender_id"], "key": "answer_length",
                "content": "技术问题喜欢简短回答，先给结论", "evidence_ids": [evidence["message_id"]],
                "confidence": 0.95}]}))
        self.host.llm_generate = model
        await self.plugin.observe_message(Event(text="技术问题回答短一点，先给结论"))
        self.plugin.backend.engine.delay = 0
        await self.plugin.backend.engine.run_once()
        request = SimpleNamespace(system_prompt="", extra_user_content_parts=[])
        await self.plugin.inject_memory(Event(mid="m2", name="新的昵称", text="这个报错怎么处理"), request)
        self.assertIn("技术问题喜欢简短回答", request.extra_user_content_parts[0].text)
        self.assertTrue(request.extra_user_content_parts[0]._no_save)

    async def test_cannot_store_data_in_plugin_source_directory(self):
        plugin = self.module.ChuandaMemoryPlugin(self.host, {"data_dir": str(Path(self.module.__file__).parent / "data")})
        with self.assertRaises(ValueError):
            await plugin.initialize()
