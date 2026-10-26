import json
import sqlite3
import tempfile
import unittest

from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore
from tests.support import temporary_directory


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_800_000_000.0
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)

    def observe(self, sender="1001", text="我喜欢 Rust", *, ns="qq-bot1", scope="group:1", mid="1", name="小王"):
        return self.store.observe(Observation(ns, scope, mid, sender, name, text))

    def candidate(self, **kwargs):
        return dict(kind="preference", subject_id="1001", key="language", content="喜欢 Rust", evidence_ids=["1"],
                    participant_ids=[], confidence=0.95, ttl_days=None) | kwargs

    def write(self, candidate=None):
        batch = self.store.next_batch(delay=0, limit=16)
        return self.store.apply_batch(batch, [candidate or self.candidate()])

    def profile(self, person="1001", scope="group:1", ns="qq-bot1"):
        return self.store.profile(ns, scope, person)

    def test_duplicate_delivery_does_not_duplicate_or_rename(self):
        self.assertTrue(self.observe())
        self.assertFalse(self.observe(name="伪重放"))
        self.assertEqual(self.profile()["name"], "小王")

    def test_identity_survives_nickname_change_and_reopen(self):
        self.observe()
        self.write()
        self.observe(mid="2", name="期末不挂科")
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)
        profile = self.profile()
        self.assertEqual(profile["person_id"], "1001")
        self.assertEqual(profile["aliases"], ["小王", "期末不挂科"])
        self.assertEqual(len(profile["memories"]), 1)

    def test_same_nickname_does_not_merge_people(self):
        self.observe()
        self.observe(sender="1002", mid="2")
        self.write()
        self.assertEqual(len(self.profile("1002")["memories"]), 0)

    def test_private_group_and_other_bot_are_isolated(self):
        self.observe()
        self.write()
        for scope, ns in [("private:1001", "qq-bot1"), ("group:2", "qq-bot1"), ("group:1", "qq-bot2")]:
            self.assertFalse(self.profile(scope=scope, ns=ns)["memories"])
            self.assertFalse(self.store.context(ns, scope, "1001", "Rust")["memories"])

    def test_correction_updates_same_memory_id(self):
        self.observe()
        self.write()
        before = self.profile()["memories"][0]["id"]
        self.observe(mid="2", text="纠正一下，我现在喜欢 Go")
        self.now += 60
        self.write(self.candidate(content="现在喜欢 Go", evidence_ids=["2"]))
        memories = self.profile()["memories"]
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], before)
        self.assertEqual(memories[0]["content"], "现在喜欢 Go")

    def test_rejects_third_party_profile_and_fake_evidence(self):
        self.observe(sender="1002", text="小王喜欢 Rust")
        batch = self.store.next_batch(delay=0, limit=16)
        self.assertEqual(self.store.apply_batch(batch, [self.candidate(), self.candidate(evidence_ids=["fake"])]), (0, 2))

    def test_rejects_unobserved_group_participant(self):
        self.observe()
        result = self.write(self.candidate(kind="joke", subject_id="@group", participant_ids=["stranger"]))
        self.assertEqual(result, (0, 1))

    def test_temporary_states_expire(self):
        self.observe(text="最近在准备比赛")
        self.write(self.candidate(kind="state", content="正在准备比赛", ttl_days=1))
        self.now += 86401
        self.assertFalse(self.profile()["memories"])
        self.assertFalse(self.store.context("qq-bot1", "group:1", "1001", "比赛")["memories"])

    def test_joke_cooldown(self):
        self.observe(text="最后一把")
        self.write(self.candidate(kind="joke", subject_id="@group", content="最后一把是个熬夜梗"))
        self.assertEqual(len(self.store.context("qq-bot1", "group:1", "1001", "最后一把")["memories"]), 1)
        self.assertFalse(self.store.context("qq-bot1", "group:1", "1001", "最后一把")["memories"])
        self.now += 21601
        self.assertTrue(self.store.context("qq-bot1", "group:1", "1001", "最后一把")["memories"])

    def test_forget_during_extraction_cannot_resurrect(self):
        self.observe()
        self.write()
        memory = self.profile()["memories"][0]
        self.observe(mid="2")
        stale = self.store.next_batch(delay=0, limit=16)
        self.assertTrue(self.store.forget("qq-bot1", "group:1", "1001", memory["id"]))
        self.assertEqual(self.store.apply_batch(stale, [self.candidate(evidence_ids=["2"])]), (0, 1))
        self.observe(mid="3")
        self.assertEqual(self.write(self.candidate(evidence_ids=["3"])), (0, 1))
        self.assertFalse(self.profile()["memories"])

    def test_forget_does_not_allow_other_users(self):
        self.observe()
        self.write()
        mid = self.profile()["memories"][0]["id"]
        with self.assertRaises(PermissionError):
            self.store.forget("qq-bot1", "group:1", "1002", mid)
        self.assertFalse(self.store.forget("qq-bot1", "group:2", "1001", mid, admin=True))

    def test_clear_erases_shared_related_memory_and_keeps_other_person(self):
        self.observe()
        self.observe(sender="1002", mid="2", name="小李")
        batch = self.store.next_batch(delay=0, limit=16)
        self.store.apply_batch(batch, [self.candidate(), self.candidate(subject_id="1002", evidence_ids=["2"]),
                                      self.candidate(kind="event", subject_id="@group", evidence_ids=["1", "2"])])
        self.observe(mid="3")
        stale = self.store.next_batch(delay=0, limit=16)
        self.store.pause("qq-bot1", "group:1", "1001", erase=True)
        self.store.apply_batch(stale, [self.candidate(evidence_ids=["3"])])
        self.assertFalse(self.profile()["memories"])
        self.assertEqual(self.profile()["aliases"], [])
        self.assertTrue(self.profile()["paused"])
        self.assertEqual(len(self.profile("1002")["memories"]), 1)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memories WHERE subject_id='@group'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages WHERE sender_id='1001'").fetchone()[0], 0)
        self.assertFalse(self.observe(mid="4"))
        self.store.resume("qq-bot1", "group:1", "1001")
        self.assertTrue(self.observe(mid="5"))

    def test_backup_is_readable(self):
        self.observe()
        self.write()
        backup = self.store.backup()
        db = sqlite3.connect(backup)
        self.addCleanup(db.close)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)

    def test_newer_database_is_not_downgraded(self):
        with self.store.connection() as db:
            db.execute("PRAGMA user_version=999")
        with self.assertRaises(RuntimeError):
            MemoryStore(self.temp.name)
        with self.store.connection() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 999)

    def test_prune_preserves_pending_and_long_term_memory(self):
        self.observe()
        self.write()
        self.observe(mid="2")
        self.now += 15 * 86400
        self.store.prune()
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT message_id FROM messages").fetchone()[0], "2")
        self.assertTrue(self.profile()["memories"])

    def test_rejects_nonfinite_confidence_and_unsupported_schema(self):
        self.observe()
        batch = self.store.next_batch(delay=0, limit=16)
        candidates = [self.candidate(confidence=float("nan")), self.candidate(ttl_days=float("inf")),
                      self.candidate(kind="instruction"), self.candidate(confidence=True)]
        self.assertEqual(self.store.apply_batch(batch, candidates), (0, 4))

    def test_retrieval_budget_and_unrelated_other_person(self):
        self.observe()
        self.write()
        self.assertFalse(self.store.context("qq-bot1", "group:1", "1002", "晚饭")["memories"])
        self.assertFalse(self.store.context("qq-bot1", "group:1", "1001", "Rust", char_budget=1)["memories"])

    def test_other_person_can_be_recalled_by_name_without_merging_ids(self):
        self.observe()
        self.write()
        recalled = self.store.context("qq-bot1", "group:1", "1002", "小王平常用什么写代码")
        self.assertEqual(recalled["memories"][0]["subject_id"], "1001")
