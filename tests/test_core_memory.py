import json
import sqlite3
import unittest
from contextlib import closing

from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


class CoreMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_800_000_000.0
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)

    def observe(self, mid="a1", person="A", text="我参与了那次通宵修好比赛系统的事"):
        self.store.observe(Observation("qq", "group:1", mid, person, person + "同学", text))

    def apply(self, **changes):
        batch = self.store.next_batch(delay=0, limit=20)
        return self.store.apply_batch(batch, [dict(kind="event", subject_id="@group", key="contest_rescue",
            content="大家通宵修好比赛系统，成了难忘的共同经历", evidence_ids=[batch["messages"][0]["message_id"]],
            confidence=0.95, importance=5, retention="core") | changes])

    def memories(self, person="@group"):
        return self.store.profile("qq", "group:1", person)["memories"]

    def test_single_impression_and_same_day_flood_are_not_permanent(self):
        for index in range(5):
            self.observe(str(index), "A" if index % 2 else "B")
            self.apply()
        memory = self.memories()[0]
        self.assertFalse(memory["core"])
        self.assertEqual(len(memory["evidence_days"]), 1)
        self.assertIsNotNone(memory["expires_at"])

    def test_spaced_group_reinforcement_promotes_and_survives_expiry_and_cap(self):
        for index in range(3):
            self.observe(str(index), "A" if index % 2 else "B")
            self.apply()
            self.now += 86400
        original = self.memories()[0]
        self.assertTrue(original["core"])
        self.assertIsNone(original["expires_at"])
        self.store.max_group_memories = 1
        self.observe("new")
        self.apply(key="ordinary", content="一次普通的小组交流", importance=3, retention="normal")
        self.now += 400 * 86400
        self.store.prune()
        self.assertEqual([m["id"] for m in self.memories()], [original["id"]])

    def test_delayed_single_batch_uses_actual_receipt_days(self):
        for index in range(3):
            self.observe(str(index), "A" if index % 2 else "B")
            self.now += 86400
        self.apply(evidence_ids=["0", "1", "2"])
        self.assertTrue(self.memories()[0]["core"])

    def test_same_day_sources_processed_on_later_days_cannot_fake_spaced_evidence(self):
        for index in range(3):
            self.observe(str(index), "A" if index % 2 else "B")
        self.now += 10 * 86400
        self.apply(evidence_ids=["0", "1", "2"])
        self.assertFalse(self.memories()[0]["core"])

    def test_one_speaker_group_repetition_does_not_promote(self):
        for index in range(3):
            self.observe(str(index))
            self.apply()
            self.now += 86400
        self.assertFalse(self.memories()[0]["core"])

    def test_explicit_personal_request_and_correction_preserve_core_identity(self):
        self.observe(text="永久记住：我最看重朋友之间相互帮忙")
        self.apply(kind="preference", subject_id="A", content="最看重朋友之间相互帮忙")
        original = self.memories("A")[0]
        self.assertTrue(original["core"])
        self.observe("a2", text="更正一下，我最看重的是相互尊重")
        self.apply(kind="preference", subject_id="A", content="最看重朋友之间相互尊重", retention="normal")
        current = self.memories("A")[0]
        self.assertEqual(current["id"], original["id"])
        self.assertTrue(current["core"])
        self.assertIn("尊重", current["content"])

    def test_negated_permanent_request_does_not_promote(self):
        self.observe(text="不要永久记住这次比赛的小事")
        self.apply(kind="fact", subject_id="A", content="参与比赛")
        self.assertFalse(self.memories("A")[0]["core"])

    def test_temporary_state_cannot_be_permanent_even_on_explicit_request(self):
        self.observe(text="永久记住我今天有点累")
        self.apply(kind="state", subject_id="A", content="今天有点累")
        self.assertFalse(self.memories("A")[0]["core"])
        with self.assertRaises(ValueError):
            self.store.pin("qq", "group:1", "A", self.memories("A")[0]["id"])

    def test_pin_permissions_unpin_and_manual_release_survive_model_updates(self):
        self.observe()
        self.apply()
        memory_id = self.memories()[0]["id"]
        with self.assertRaises(PermissionError):
            self.store.pin("qq", "group:1", "A", memory_id)
        self.assertFalse(self.store.pin("qq", "group:2", "A", memory_id, admin=True))
        self.assertTrue(self.store.pin("qq", "group:1", "A", memory_id, admin=True))
        self.assertTrue(self.memories()[0]["core"])
        self.store.pin("qq", "group:1", "A", memory_id, permanent=False, admin=True)
        for index in range(3):
            self.now += 86400
            self.observe(str(index), "B" if index % 2 else "A")
            self.apply()
        self.assertFalse(self.memories()[0]["core"])
        self.assertIsNotNone(self.memories()[0]["expires_at"])

    def test_pinning_invalidates_stale_work_without_dropping_queue(self):
        self.observe()
        self.apply(kind="fact", subject_id="A")
        memory_id = self.memories("A")[0]["id"]
        self.observe("next")
        batch = self.store.next_batch(delay=0, limit=20)
        self.store.pin("qq", "group:1", "A", memory_id)
        self.assertEqual(self.store.apply_batch(batch, []), (0, 0))
        self.assertEqual(self.store.profile("qq", "group:1", "A")["stats"]["pending"], 1)

    def test_capacity_uses_importance_before_recency(self):
        self.store.max_group_memories = 1
        self.observe()
        self.apply(retention="normal")
        original = self.memories()[0]["id"]
        self.now += 10
        self.observe("next")
        self.apply(key="trivia", content="今天看了个普通帖子", importance=1, retention="normal")
        self.assertEqual([m["id"] for m in self.memories()], [original])

    def test_forget_core_removes_metadata_and_still_blocks_recreation(self):
        self.observe(text="永久记住我爱写编译器")
        self.apply(kind="preference", subject_id="A", content="喜欢写编译器")
        memory_id = self.memories("A")[0]["id"]
        self.store.forget("qq", "group:1", "A", memory_id)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_policy").fetchone()[0], 0)
        self.observe("again", text="永久记住我爱写编译器")
        self.assertEqual(self.apply(kind="preference", subject_id="A", content="喜欢写编译器"), (0, 1))

    def test_legacy_v1_rows_and_custom_policy_are_untouched_by_additive_metadata(self):
        self.observe()
        self.apply()
        self.store.prompt_path.write_text("我自己的策略", encoding="utf-8")
        with self.store.connection() as db:
            original = tuple(db.execute("SELECT * FROM memories").fetchone())
            db.executescript("DROP TRIGGER memories_remove_policy; DROP VIEW memory_details; DROP TABLE memory_policy;")
        reopened = MemoryStore(self.temp.name, clock=lambda: self.now)
        with reopened.connection() as db:
            self.assertEqual(tuple(db.execute("SELECT * FROM memories").fetchone()), original)
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertEqual(reopened.prompt_path.read_text(encoding="utf-8"), "我自己的策略")

    def test_readable_view_and_backup_include_core_policy(self):
        self.observe()
        self.apply()
        self.store.pin("qq", "group:1", "A", self.memories()[0]["id"], admin=True)
        self.store.export_readable()
        data = json.loads((self.store.directory / "memory.json").read_text(encoding="utf-8"))
        self.assertTrue(data["scopes"][0]["group_memories"][0]["core"])
        self.assertIn("永久核心", (self.store.directory / "memory.md").read_text(encoding="utf-8"))
        with closing(sqlite3.connect(self.store.automatic_backup())) as db:
            self.assertEqual(db.execute("SELECT core FROM memory_policy").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
