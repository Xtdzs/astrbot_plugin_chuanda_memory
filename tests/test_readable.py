import json
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


class ReadableTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_800_000_000.0
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)

    def read(self):
        return json.loads((self.store.directory / "memory.json").read_text(encoding="utf-8"))

    def text(self):
        return (self.store.directory / "memory.md").read_text(encoding="utf-8")

    def observe(self, **kwargs):
        self.store.observe(Observation("qq", "group:1", kwargs.get("mid", "m1"), "1001", kwargs.get("name", "小王"),
                                       "原始聊天正文不应复制到快照"))

    def write(self, **kwargs):
        batch = self.store.next_batch(delay=0, limit=16)
        return self.store.apply_batch(batch, [dict(kind="preference", subject_id="1001", key="answer_length",
            content="技术问题喜欢先给结论", evidence_ids=["m1"], confidence=0.95) | kwargs])

    def test_views_are_created_automatically_and_include_no_raw_chat(self):
        self.assertEqual(self.read()["scopes"], [])
        self.observe()
        self.write()
        data = self.read()
        person = data["scopes"][0]["people"][0]
        self.assertEqual(person["person_id"], "1001")
        self.assertEqual(person["memories"][0]["content"], "技术问题喜欢先给结论")
        self.assertEqual(person["memories"][0]["evidence_ids"], ["m1"])
        self.assertIn("小王", self.text())
        self.assertIn("技术问题喜欢先给结论", self.text())
        self.assertIn(data["generation"], self.text())
        self.assertNotIn("原始聊天正文", self.text())
        self.assertNotIn("原始聊天正文", json.dumps(data, ensure_ascii=False))

    def test_correction_nickname_and_forget_refresh_both_views(self):
        self.observe()
        self.write()
        mid = self.read()["scopes"][0]["people"][0]["memories"][0]["id"]
        self.observe(mid="m2", name="新的昵称")
        self.write(content="希望先给示例", evidence_ids=["m2"])
        self.assertIn("新的昵称", self.text())
        self.assertNotIn("技术问题喜欢先给结论", self.text())
        self.assertTrue(self.store.forget("qq", "group:1", "1001", mid))
        self.assertNotIn("希望先给示例", self.text())
        self.assertEqual(self.read()["scopes"][0]["people"][0]["memories"], [])
        self.assertEqual(self.read()["scopes"][0]["blocked_topics"][0]["key"], "answer_length")

    def test_clear_pause_resume_and_expiry_refresh(self):
        self.observe()
        self.write(kind="state", ttl_days=1)
        self.now += 86401
        self.store.prune()
        self.assertNotIn("技术问题喜欢先给结论", self.text())
        self.store.pause("qq", "group:1", "1001", erase=True)
        person = self.read()["scopes"][0]["people"][0]
        self.assertEqual(person["name"], "")
        self.assertEqual(person["aliases"], [])
        self.assertTrue(person["paused"])
        self.assertNotIn("小王", self.text())
        self.store.resume("qq", "group:1", "1001")
        self.assertFalse(self.read()["scopes"][0]["people"][0]["paused"])

    def test_views_rebuild_from_existing_database_without_importing_file_edits(self):
        self.observe()
        self.write()
        (self.store.directory / "memory.json").write_text('{"forged":"不要写进数据库"}', encoding="utf-8")
        (self.store.directory / "memory.md").unlink()
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)
        self.assertIn("技术问题喜欢先给结论", self.text())
        self.assertNotIn("forged", self.read())
        self.assertEqual(self.store.profile("qq", "group:1", "1001")["memories"][0]["content"], "技术问题喜欢先给结论")

    def test_failed_atomic_replace_preserves_previous_file_and_db_commit(self):
        before = (self.store.directory / "memory.json").read_bytes()
        with patch("chuanda_memory.readable.os.replace", side_effect=PermissionError("locked")):
            with self.assertLogs("chuanda_memory.store", level="WARNING"):
                self.observe()
        self.assertEqual((self.store.directory / "memory.json").read_bytes(), before)
        self.assertEqual(self.store.profile("qq", "group:1", "1001")["name"], "小王")
        self.assertFalse(list(self.store.directory.glob(".cdmem-*.tmp")))
        self.store.export_readable()
        self.assertEqual(self.read()["scopes"][0]["people"][0]["name"], "小王")

    def test_group_memories_and_markdown_escaping(self):
        self.observe(name="<script>坏名字</script>")
        self.write(kind="joke", subject_id="@group", content="最后一把：后来又打了两小时")
        self.assertEqual(len(self.read()["scopes"][0]["group_memories"]), 1)
        self.assertIn("最后一把", self.text())
        self.assertNotIn("<script>", self.text())
        self.assertIn("&lt;script&gt;", self.text())

    def test_failed_transaction_does_not_update_views(self):
        before = (self.store.directory / "memory.json").read_bytes()
        with self.assertRaises(RuntimeError):
            with self.store.connection(sync_readable=True) as db:
                db.execute("INSERT INTO scopes VALUES('bad','bad',0)")
                raise RuntimeError("rollback")
        self.assertEqual((self.store.directory / "memory.json").read_bytes(), before)
        self.assertEqual(self.read()["scopes"], [])

    def test_background_export_cannot_block_message_commit(self):
        entered, release = threading.Event(), threading.Event()
        from chuanda_memory.readable import write_views
        def blocked(*args):
            entered.set()
            release.wait(timeout=5)
            write_views(*args)
        with patch("chuanda_memory.store.write_views", side_effect=blocked):
            store = MemoryStore(self.temp.name, readable_background=True)
            try:
                self.assertTrue(entered.wait(timeout=2))
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(store.observe, Observation("qq", "group:1", "async", "1001", "小王", "你好"))
                    self.assertTrue(future.result(timeout=2))
                self.assertFalse(release.is_set())
            finally:
                release.set()
                store.close()
        self.assertEqual(self.read()["scopes"][0]["people"][0]["person_id"], "1001")
