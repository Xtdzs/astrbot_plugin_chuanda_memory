import json
import unittest

from chuanda_memory.models import Observation
from chuanda_memory.store import MemoryStore, same_content
from tests.support import temporary_directory


class PersonBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_800_000_000.0
        self.store = MemoryStore(self.temp.name, clock=lambda: self.now)

    def observe(self, sender="A", mid="m1", text="我喜欢 Rust", scope="group:1"):
        self.store.observe(Observation("qq", scope, mid, sender, "成员" + sender, text))

    def batch(self):
        return self.store.next_batch(delay=300, limit=20, mode="person", min_batch=3, max_wait=1800)

    def item(self, **changes):
        return dict(kind="preference", subject_id="A", key="language", content="喜欢 Rust", evidence_ids=["m1"],
                    confidence=0.95) | changes

    def apply(self, item):
        batch = self.store.next_batch(delay=0, limit=40)
        return self.store.apply_batch(batch, [item])

    def memories(self, subject="A"):
        return self.store.profile("qq", "group:1", subject)["memories"]

    def test_twenty_messages_are_counted_per_person(self):
        for i in range(20):
            self.observe("A", "a" + str(i))
        for i in range(19):
            self.observe("B", "b" + str(i))
        batch = self.batch()
        self.assertEqual(batch["target_id"], "A")
        self.assertEqual(len(batch["messages"]), 20)
        self.assertEqual({m["sender_id"] for m in batch["messages"]}, {"A"})
        self.store.apply_batch(batch, [])
        self.assertIsNone(self.batch())
        self.assertEqual(self.store.profile("qq", "group:1", "B")["stats"]["pending"], 19)

    def test_low_volume_is_processed_after_max_wait(self):
        self.observe()
        self.now += 301
        self.assertIsNone(self.batch())
        self.now += 1500
        self.assertEqual(len(self.batch()["messages"]), 1)

    def test_three_messages_can_flush_after_idle_period(self):
        for i in range(3):
            self.observe(mid=str(i))
        self.now += 299
        self.assertIsNone(self.batch())
        self.now += 2
        self.assertEqual(len(self.batch()["messages"]), 3)

    def test_explicit_remember_and_correction_do_not_wait_for_twenty(self):
        self.observe(text="记住：技术问题先给结论")
        self.assertEqual(len(self.batch()["messages"]), 1)

    def test_group_context_does_not_consume_another_persons_queue(self):
        self.observe("A", "a", "记住：我喜欢简短回答")
        self.observe("B", "b", "我也喜欢大家简短地聊技术")
        batch = self.batch()
        self.assertEqual(batch["target_id"], "A")
        self.assertIn("b", {m["message_id"] for m in batch["recent_context"]})
        group = self.item(kind="group_profile", subject_id="@group", key="conversation_style",
                          content="近期可见讨论里，两位成员都表达了偏好简短的技术交流", evidence_ids=["a", "b"])
        self.assertEqual(self.store.apply_batch(batch, [group]), (1, 0))
        self.assertEqual(self.store.profile("qq", "group:1", "B")["stats"]["pending"], 1)
        recalled = self.store.context("qq", "group:1", "A", "你好")
        self.assertEqual(recalled["memories"][0]["kind"], "group_profile")

    def test_person_batch_cannot_write_another_persons_profile(self):
        self.observe("A", "a", "记住：我喜欢短回答")
        self.observe("B", "b", "我喜欢 Rust")
        batch = self.batch()
        self.assertEqual(self.store.apply_batch(batch, [self.item(subject_id="B", evidence_ids=["b"])]), (0, 1))

    def test_group_profile_requires_two_speakers(self):
        self.observe()
        self.assertEqual(self.apply(self.item(kind="group_profile", subject_id="@group")), (0, 1))

    def test_style_requires_three_distinct_self_messages_and_expires(self):
        self.observe(mid="m1")
        self.observe(mid="m2")
        self.observe(mid="m3")
        batch = self.store.next_batch(delay=0, limit=40)
        self.assertEqual(self.store.apply_batch(batch, [self.item(kind="style", evidence_ids=["m1", "m1", "m2"])]), (0, 1))
        self.observe(mid="m4")
        batch = self.store.next_batch(delay=0, limit=40, mode="person")
        style = self.item(kind="style", content="近期可见聊天中习惯简短表达", evidence_ids=["m1", "m2", "m4"], ttl_days=365)
        self.assertEqual(self.store.apply_batch(batch, [style]), (1, 0))
        self.now += 31 * 86400
        self.assertFalse(self.memories())

    def test_same_content_different_key_is_merged_and_evidence_accumulates(self):
        self.observe()
        self.apply(self.item())
        original = self.memories()[0]["id"]
        self.observe(mid="m2")
        self.apply(self.item(key="favorite_programming_language", content="喜欢 Ｒｕｓｔ！", evidence_ids=["m2"]))
        memories = self.memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], original)
        self.assertEqual(memories[0]["evidence_ids"], ["m1", "m2"])

    def test_key_prefix_and_case_do_not_duplicate_the_same_topic(self):
        self.observe()
        self.apply(self.item(key="preference.answer_length", content="喜欢短回答"))
        self.observe(mid="m2")
        self.apply(self.item(key="ANSWER.LENGTH", content="希望回答更详细", evidence_ids=["m2"]))
        self.assertEqual(len(self.memories()), 1)
        self.assertEqual(self.memories()[0]["content"], "希望回答更详细")

    def test_explicit_update_id_handles_a_semantic_rewrite(self):
        self.observe()
        self.apply(self.item(content="平时会用 Rust 写小工具"))
        original = self.memories()[0]["id"]
        self.observe(mid="m2")
        self.apply(self.item(key="tools", content="写一些小工具时常用 Rust", evidence_ids=["m2"], update_id=original))
        self.assertEqual(len(self.memories()), 1)
        self.assertEqual(self.memories()[0]["id"], original)

    def test_update_id_cannot_cross_a_person_boundary(self):
        self.observe()
        self.apply(self.item())
        original = self.memories()[0]["id"]
        self.observe("B", "b")
        self.assertEqual(self.apply(self.item(subject_id="B", evidence_ids=["b"], update_id=original)), (0, 1))

    def test_fuzzy_matching_preserves_negation_and_numbers(self):
        self.assertFalse(same_content("以后技术问题希望详细回答并附带三个示例", "以后技术问题不希望详细回答并附带三个示例"))
        self.assertFalse(same_content("计划在2026年10月参加这场程序设计比赛", "计划在2026年11月参加这场程序设计比赛"))
        self.assertFalse(same_content("喜欢 C#", "喜欢 C"))
        self.assertFalse(same_content("使用 Python 3.10", "使用 Python 310"))

    def test_deleted_content_cannot_return_under_a_different_key(self):
        self.observe()
        self.apply(self.item())
        self.store.forget("qq", "group:1", "A", self.memories()[0]["id"])
        self.observe(mid="m2")
        self.assertEqual(self.apply(self.item(key="new_name", content="喜欢 Ｒｕｓｔ！", evidence_ids=["m2"])), (0, 1))

    def test_person_capacity_preserves_explicit_boundaries(self):
        self.store.max_person_memories = 2
        self.observe()
        batch = self.store.next_batch(delay=0, limit=40)
        candidates = [self.item(key="topic" + str(i), content="不同的兴趣" + str(i)) for i in range(4)]
        candidates += [self.item(kind="boundary", key="name", content="不要叫我某个外号")]
        self.store.apply_batch(batch, candidates)
        memories = self.memories()
        self.assertEqual(len([m for m in memories if m["kind"] != "boundary"]), 2)
        self.assertEqual(len([m for m in memories if m["kind"] == "boundary"]), 1)

    def test_group_profile_deletion_requires_admin(self):
        self.observe("A", "a")
        self.observe("B", "b")
        self.apply(self.item(kind="group_profile", subject_id="@group", evidence_ids=["a", "b"]))
        memory_id = self.memories("@group")[0]["id"]
        with self.assertRaises(PermissionError):
            self.store.forget("qq", "group:1", "A", memory_id)
        self.assertTrue(self.store.forget("qq", "group:1", "A", memory_id, admin=True))

    def test_group_capacity_prioritizes_group_profile_over_old_events(self):
        self.store.max_group_memories = 2
        self.observe("A", "a")
        self.observe("B", "b")
        batch = self.store.next_batch(delay=0, limit=40)
        profile = self.item(kind="group_profile", subject_id="@group", key="group_topics", content="近期共同讨论技术", evidence_ids=["a", "b"])
        events = [self.item(kind="event", subject_id="@group", key="event" + str(i), content="不同共同事件" + str(i), evidence_ids=["a"]) for i in range(3)]
        self.store.apply_batch(batch, [profile] + events)
        memories = self.memories("@group")
        self.assertEqual(len(memories), 2)
        self.assertIn("group_profile", {m["kind"] for m in memories})

    def test_summary_policy_is_editable_and_empty_means_contract_only(self):
        self.assertTrue(self.store.prompt_path.exists())
        self.store.prompt_path.write_text("仅保留我指定的交流偏好", encoding="utf-8")
        self.assertIn("仅保留我指定的交流偏好", self.store.summary_prompt())
        self.assertIn("subject_id", self.store.summary_prompt())
        self.store.prompt_path.unlink()
        self.assertIn("少发言不等于", self.store.summary_prompt())
        self.store.prompt_path.write_text("", encoding="utf-8")
        self.assertNotIn("仅保留我指定的交流偏好", self.store.summary_prompt())
        self.assertIn("subject_id", self.store.summary_prompt())

    def test_status_separates_queue_and_memories_and_reports_empty_extraction(self):
        self.observe(text="记住：我喜欢 Rust")
        self.assertEqual(self.store.profile("qq", "group:1", "A")["stats"]["pending"], 1)
        self.store.apply_batch(self.batch(), [])
        profile = self.store.profile("qq", "group:1", "A")
        self.assertEqual(profile["stats"]["processed"], 1)
        self.assertEqual(profile["stats"]["pending"], 0)
        self.assertEqual(profile["stats"]["last_summary"]["outcome"], "no_candidates")
        self.assertEqual(len(profile["memories"]), 0)

    def test_rejected_and_error_runs_are_distinguishable(self):
        self.observe(text="记住：我喜欢 Rust")
        self.store.apply_batch(self.batch(), [self.item(evidence_ids=["fake"])])
        self.assertEqual(self.store.profile("qq", "group:1", "A")["stats"]["last_summary"]["outcome"], "validation_rejected")
        self.observe(mid="m2", text="更正：请回答短一点")
        self.store.fail_batch(self.batch(), "TimeoutError")
        last = self.store.profile("qq", "group:1", "A")["stats"]["last_summary"]
        self.assertEqual(last["outcome"], "error")
        self.assertEqual(last["error_type"], "TimeoutError")

    def test_upgrade_adds_auxiliary_state_without_rewriting_existing_memories(self):
        self.observe()
        self.apply(self.item())
        original = self.memories()[0]["id"]
        with self.store.connection() as db:
            db.execute("DROP TABLE summary_runs")
        self.store.prompt_path.unlink()
        reopened = MemoryStore(self.temp.name, clock=lambda: self.now)
        self.assertEqual(reopened.profile("qq", "group:1", "A")["memories"][0]["id"], original)
        self.assertTrue(reopened.prompt_path.exists())
        self.assertEqual(reopened.profile("qq", "group:1", "A")["stats"]["retained"], 1)
