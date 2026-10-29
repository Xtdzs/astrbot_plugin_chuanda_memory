import json
import sqlite3
import unittest
import threading
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from chuanda_memory.extraction import extraction_prompt
from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


class DurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_800_000_000.0
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)

    def observe(self, person="A", mid="a1", text="我喜欢 Rust", name=None):
        return self.store.observe(Observation("qq", "group:1", mid, person, name or (person + "同学"), text))

    def batch(self):
        return self.store.next_batch(delay=0, limit=20, mode="person")

    def item(self, person="A", mid="a1", **changes):
        return dict(kind="preference", subject_id=person, key="language", content="喜欢 Rust",
                    evidence_ids=[mid], confidence=0.95) | changes

    def profile(self, person="A"):
        return self.store.profile("qq", "group:1", person)

    def test_pause_preserves_other_queue_and_invalidates_inflight_results(self):
        self.observe()
        self.observe("B", "b1")
        stale = self.store.next_batch(delay=0, limit=20, mode="legacy")
        self.store.pause("qq", "group:1", "A")
        self.assertEqual(self.profile()["stats"]["pending"], 0)
        self.assertEqual(self.profile("B")["stats"]["pending"], 1)
        self.assertEqual(self.store.apply_batch(stale, [self.item("B", "b1")]), (0, 1))
        self.assertEqual(self.batch()["target_id"], "B")
        self.assertEqual(self.store.apply_batch(self.batch(), [self.item("B", "b1")]), (1, 0))

    def test_forget_preserves_other_queue_without_allowing_resurrection(self):
        self.observe()
        self.store.apply_batch(self.batch(), [self.item()])
        memory_id = self.profile()["memories"][0]["id"]
        self.observe(mid="a2")
        self.observe("B", "b1")
        stale = self.store.next_batch(delay=0, limit=20, mode="legacy")
        self.store.forget("qq", "group:1", "A", memory_id)
        self.assertEqual(self.profile("B")["stats"]["pending"], 1)
        self.assertEqual(self.profile()["stats"]["pending"], 0)
        self.assertEqual(self.store.apply_batch(stale, [self.item("B", "b1")]), (0, 1))
        self.store.apply_batch(self.batch(), [self.item("B", "b1")])
        self.observe(mid="a3")
        self.assertEqual(self.store.apply_batch(self.batch(), [self.item(mid="a3", key="new_name")]), (0, 1))
        self.assertFalse(self.profile()["memories"])

    def test_retries_continue_after_three_failures_and_back_off(self):
        self.observe()
        previous_delay = 0
        for attempt in range(12):
            batch = self.batch()
            self.assertIsNotNone(batch)
            self.store.fail_batch(batch, "TimeoutError")
            with self.store.connection() as db:
                row = db.execute("SELECT attempts,retry_at FROM messages").fetchone()
            self.assertEqual(row["attempts"], attempt + 1)
            delay = row["retry_at"] - self.now
            self.assertGreaterEqual(delay, previous_delay)
            self.assertLessEqual(delay, 21600)
            self.assertIsNone(self.batch())
            self.now = row["retry_at"] + 1
            previous_delay = delay
        self.assertEqual(self.store.apply_batch(self.batch(), [self.item()]), (1, 0))

    def test_legacy_failed_queue_is_recoverable_and_not_pruned(self):
        self.observe()
        with self.store.connection() as db:
            db.execute("UPDATE messages SET attempts=3,retry_at=?", (self.now + 30,))
        self.now += 15 * 86400
        self.store.prune()
        self.assertEqual(self.profile()["stats"]["failed"], 1)
        self.assertIsNotNone(self.batch())

    def test_manual_retry_is_scoped_and_keeps_processed_messages_done(self):
        self.observe()
        self.observe("B", "b1")
        with self.store.connection() as db:
            db.execute("UPDATE messages SET attempts=4,retry_at=?", (self.now + 1000,))
        self.assertEqual(self.store.retry_pending("qq", "group:1", "A"), 1)
        self.store.apply_batch(self.batch(), [])
        self.assertEqual(self.store.retry_pending("qq", "group:1", "A"), 0)
        self.assertIsNone(self.batch())
        self.assertEqual(self.profile("B")["stats"]["retrying"], 1)

    def test_identical_evidence_cannot_renew_or_rewrite_memory(self):
        self.observe()
        batch = self.batch()
        self.store.apply_batch(batch, [self.item(kind="state", ttl_days=7)])
        original = self.profile()["memories"][0]
        self.now += 86400
        self.assertEqual(self.store.apply_batch(batch, [self.item(kind="state", ttl_days=30, content="改写的内容")]), (0, 0))
        current = self.profile()["memories"][0]
        for field in ("id", "content", "updated_at", "expires_at"):
            self.assertEqual(current[field], original[field])
        self.assertEqual(self.profile()["stats"]["last_summary"]["outcome"], "unchanged")

    def test_new_evidence_can_refresh_same_memory_id(self):
        self.observe()
        self.store.apply_batch(self.batch(), [self.item(kind="state", ttl_days=7)])
        original = self.profile()["memories"][0]
        self.now += 86400
        self.observe(mid="a2")
        self.store.apply_batch(self.batch(), [self.item(mid="a2", kind="state", ttl_days=7)])
        current = self.profile()["memories"][0]
        self.assertEqual(current["id"], original["id"])
        self.assertGreater(current["expires_at"], original["expires_at"])

    def test_recent_context_alone_cannot_recreate_group_memory(self):
        self.observe()
        self.observe("B", "b1")
        self.store.apply_batch(self.store.next_batch(delay=0, limit=20, mode="legacy"), [])
        self.observe(mid="a2", text="今天吃饭了")
        item = self.item(kind="group_profile", subject_id="@group", evidence_ids=["a1", "b1"])
        self.assertEqual(self.store.apply_batch(self.batch(), [item]), (0, 1))

    def test_group_correction_drops_old_participants_when_old_evidence_is_replaced(self):
        self.observe()
        self.observe("B", "b1")
        first = self.item(kind="group_profile", subject_id="@group", evidence_ids=["a1", "b1"], content="近期共同讨论 Rust")
        self.store.apply_batch(self.store.next_batch(delay=0, limit=20, mode="legacy"), [first])
        self.observe(mid="a2")
        self.observe("C", "c1")
        changed = self.item(kind="group_profile", subject_id="@group", evidence_ids=["a2", "c1"], content="近期共同讨论 Go")
        self.store.apply_batch(self.store.next_batch(delay=0, limit=20, mode="legacy"), [changed])
        self.assertEqual(self.profile("@group")["memories"][0]["participant_ids"], ["A", "C"])
        self.store.pause("qq", "group:1", "B", erase=True)
        self.assertTrue(self.profile("@group")["memories"])

    def test_old_relevant_memory_is_ranked_before_candidate_limit(self):
        self.observe("B", "b1", "我在维护远古编译器")
        self.store.apply_batch(self.batch(), [self.item("B", "b1", content="在维护远古编译器")])
        old_id = self.profile("B")["memories"][0]["id"]
        with self.store.connection() as db:
            for i in range(1100):
                db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                           ("filler" + str(i), "qq", "group:1", "P" + str(i % 40), "fact", "topic" + str(i),
                            "不相关的最新消息" + str(i), "[]", "[]", 0.9, self.now + i + 1, self.now + i + 1, None))
        for query in ("B同学之前在维护什么", "远古编译器是谁维护的"):
            self.assertIn(old_id, {m["id"] for m in self.store.context("qq", "group:1", "A", query)["memories"]})
        self.assertFalse(self.store.context("qq", "group:2", "A", "远古编译器")["memories"])

    def test_unchanged_name_does_not_trigger_full_export(self):
        self.observe()
        with patch.object(self.store, "_sync_readable") as notify:
            self.observe(mid="a2")
            notify.assert_not_called()
            self.observe(mid="a3", name="新昵称")
            notify.assert_called_once()

    def test_daily_backup_preserves_core_policy_and_manual_backups(self):
        self.observe()
        self.store.apply_batch(self.batch(), [self.item()])
        memory_id = self.profile()["memories"][0]["id"]
        self.store.prompt_path.write_text("自定义总结策略", encoding="utf-8")
        manual = self.store.backup()
        snapshot = self.store.automatic_backup(keep_days=2)
        before = snapshot.read_bytes()
        self.assertEqual(self.store.automatic_backup(keep_days=2), snapshot)
        self.assertEqual(snapshot.read_bytes(), before)
        with closing(sqlite3.connect(snapshot)) as db:
            self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("SELECT id FROM memories").fetchone()[0], memory_id)
        self.assertEqual(snapshot.with_suffix(".summary_prompt.txt").read_text(encoding="utf-8"), "自定义总结策略")
        self.now += 2 * 86400
        self.store.automatic_backup(keep_days=2)
        self.assertFalse(snapshot.exists())
        self.assertTrue(manual.exists())
        self.assertEqual(self.profile()["memories"][0]["id"], memory_id)

    def test_prompt_preserves_complete_facts_but_drops_storage_metadata(self):
        existing = [{"id": "m1", "subject_id": "A", "kind": "fact", "key": "topic", "content": "完整内容" * 100,
                     "namespace": "private-storage", "evidence_ids": ["source" * 200]}]
        item = json.loads(extraction_prompt([], existing))["existing_memories"][0]
        self.assertEqual(item["content"], existing[0]["content"])
        self.assertEqual(item["id"], "m1")
        self.assertNotIn("evidence_ids", item)
        self.assertNotIn("namespace", item)

    def test_automatic_backup_does_not_hold_the_message_path_lock(self):
        self.observe()
        entered, release = threading.Event(), threading.Event()
        original_connect = sqlite3.connect
        backup_thread = []
        class DelayedReader:
            def __init__(self, db): self.db = db
            def execute(self, *args): return self.db.execute(*args)
            def close(self): self.db.close()
            def backup(self, *args, **kwargs):
                entered.set()
                if not release.wait(timeout=5): raise TimeoutError("Test backup was not released")
                return self.db.backup(*args, **kwargs)
        def connect(path, *args, **kwargs):
            db = original_connect(path, *args, **kwargs)
            if backup_thread and threading.get_ident() == backup_thread[0] and path == self.store.path:
                return DelayedReader(db)
            return db
        def run_backup():
            backup_thread.append(threading.get_ident())
            return self.store.automatic_backup()
        with patch("chuanda_memory.store.sqlite3.connect", side_effect=connect), ThreadPoolExecutor(max_workers=2) as pool:
            backup = pool.submit(run_backup)
            try:
                self.assertTrue(entered.wait(timeout=2))
                added = pool.submit(self.observe, mid="during-backup")
                self.assertTrue(added.result(timeout=2))
            finally:
                release.set()
                backup.result(timeout=3)
