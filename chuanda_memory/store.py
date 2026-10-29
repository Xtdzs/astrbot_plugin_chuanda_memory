from __future__ import annotations

import json
import hashlib
import difflib
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
import unicodedata
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

from .models import Observation
from .readable import CoalescingWriter, atomic_write, snapshot, write_views
from .extraction import CONTRACT_PROMPT, DEFAULT_SUMMARY_POLICY

KINDS = {"fact", "preference", "boundary", "state", "style", "event", "joke", "group_profile"}
GROUP_KINDS = {"event", "joke", "group_profile"}
CORE_KINDS = {"fact", "preference", "event", "joke"}
logger = logging.getLogger(__name__)


def normalized_content(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", "", text)
    return "".join(c for c in text if c in "#+.%_/-" or unicodedata.category(c)[0] not in {"P", "Z", "C"})


def normalized_key(key, kind):
    key = unicodedata.normalize("NFKC", key).casefold().replace(":", ".").replace("_", ".")
    return key.removeprefix(kind + ".")


def same_content(first, second):
    a, b = normalized_content(first), normalized_content(second)
    if a == b:
        return True
    if min(len(a), len(b)) < 18:
        return False
    if min(len(a), len(b)) / max(len(a), len(b)) < 0.94:
        return False
    # Don't merge contrasting preferences or distinct dates/numbers on fuzzy similarity.
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    if re.findall(r"[a-z][a-z0-9_+#./-]*", a) != re.findall(r"[a-z][a-z0-9_+#./-]*", b):
        return False
    negatives = ("不", "没", "无", "别", "not", "never", "no")
    if [w for w in negatives if w in a] != [w for w in negatives if w in b]:
        return False
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return matcher.quick_ratio() >= 0.94 and matcher.ratio() >= 0.94


def terms(text: str) -> set[str]:
    """Small dependency-free Chinese/Latin lexical index; not an embedding model."""
    text = text.lower()
    result = set(re.findall(r"[a-z0-9_+#.]+", text))
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        result.update(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            result.add(run)
    return result


class MemoryStore:
    """Versioned SQLite store. No dependency on AstrBot, DSH, or conversation IDs."""

    def __init__(self, data_dir: str | Path, *, clock=time.time, readable_background=False,
                 max_person_memories=80, max_group_memories=160):
        self.directory = Path(data_dir).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "memory.sqlite3"
        self.clock = clock
        self.lock = threading.RLock()
        self.readable_background = readable_background
        self._readable_writer = None
        self.max_person_memories = max(1, int(max_person_memories))
        self.max_group_memories = max(1, int(max_group_memories))
        self.prompt_path = self.directory / "summary_prompt.txt"
        if not self.prompt_path.exists():
            try:
                with self.prompt_path.open("x", encoding="utf-8", newline="\n") as prompt:
                    prompt.write(DEFAULT_SUMMARY_POLICY)
            except FileExistsError:
                pass
        with self.connection(sync_readable=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("Memory database is newer than this program; refusing to downgrade")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scopes (
                    namespace TEXT, scope TEXT, epoch INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(namespace, scope));
                CREATE TABLE IF NOT EXISTS people (
                    namespace TEXT, scope TEXT, person_id TEXT, name TEXT,
                    aliases TEXT NOT NULL DEFAULT '[]', paused INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL, PRIMARY KEY(namespace, scope, person_id));
                CREATE TABLE IF NOT EXISTS messages (
                    namespace TEXT, scope TEXT, message_id TEXT, sender_id TEXT,
                    sender_name TEXT, text TEXT, provider_id TEXT, timestamp REAL,
                    received_at REAL, done INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(namespace, scope, message_id));
                CREATE INDEX IF NOT EXISTS messages_pending ON messages(done, retry_at, received_at);
                CREATE INDEX IF NOT EXISTS messages_scope_timeline ON messages(namespace,scope,received_at);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, namespace TEXT, scope TEXT, subject_id TEXT,
                    kind TEXT, key TEXT, content TEXT, evidence_ids TEXT,
                    participant_ids TEXT, confidence REAL, created_at REAL,
                    updated_at REAL, expires_at REAL, last_used REAL NOT NULL DEFAULT 0,
                    UNIQUE(namespace, scope, subject_id, kind, key));
                CREATE INDEX IF NOT EXISTS memories_scope ON memories(namespace, scope, subject_id);
                CREATE TABLE IF NOT EXISTS memory_policy (
                    memory_id TEXT PRIMARY KEY, importance INTEGER NOT NULL DEFAULT 3,
                    core INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
                    evidence_days TEXT NOT NULL DEFAULT '[]');
                CREATE VIEW IF NOT EXISTS memory_details AS
                    SELECT m.*,COALESCE(p.importance,3) AS importance,COALESCE(p.core,0) AS core,
                    COALESCE(p.pinned,0) AS pinned,COALESCE(p.evidence_days,'[]') AS evidence_days
                    FROM memories m LEFT JOIN memory_policy p ON p.memory_id=m.id;
                CREATE TRIGGER IF NOT EXISTS memories_remove_policy AFTER DELETE ON memories
                    BEGIN DELETE FROM memory_policy WHERE memory_id=OLD.id; END;
                CREATE TABLE IF NOT EXISTS blocks (
                    namespace TEXT, scope TEXT, subject_id TEXT, kind TEXT, key TEXT,
                    PRIMARY KEY(namespace, scope, subject_id, kind, key));
                CREATE TABLE IF NOT EXISTS changes (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT,
                    action TEXT, timestamp REAL);
                CREATE TABLE IF NOT EXISTS summary_runs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT, scope TEXT,
                    subject_id TEXT, finished_at REAL, messages INTEGER, written INTEGER,
                    rejected INTEGER, outcome TEXT, error_type TEXT);
                PRAGMA user_version=1;
            """)

    def summary_prompt(self):
        policy = self.prompt_path.read_text(encoding="utf-8-sig") if self.prompt_path.exists() else DEFAULT_SUMMARY_POLICY
        if len(policy) > 16000:
            raise ValueError("summary_prompt.txt exceeds 16000 characters")
        return CONTRACT_PROMPT + "\n用户配置的整理策略：\n" + policy

    @contextmanager
    def connection(self, *, sync_readable=False):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=10)
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA busy_timeout=10000")
                with db:
                    yield db
                if sync_readable:
                    self._sync_readable(db)
            finally:
                db.close()

    def _sync_readable(self, db):
        if self.readable_background:
            if self._readable_writer is None:
                self._readable_writer = CoalescingWriter(self._write_readable)
            self._readable_writer.notify()
        else:
            self._write_readable()

    def _write_readable(self):
        try:
            # WAL reader has its own connection: rendering/fsync never holds the message-path lock.
            reader = sqlite3.connect(self.path, timeout=10)
            try:
                reader.row_factory = sqlite3.Row
                reader.execute("PRAGMA query_only=ON")
                reader.execute("BEGIN")
                data = snapshot(reader, self.clock())
            finally:
                reader.close()
            write_views(self.directory, data)
        except (OSError, sqlite3.Error) as exc:
            # The DB commit already succeeded. Retry on the next update or restart.
            logger.warning("Chuanda readable memory sync deferred (%s)", type(exc).__name__)

    def export_readable(self):
        """Rebuild both views from SQLite; never imports user edits into memory."""
        if self._readable_writer:
            self._readable_writer.flush()
        else:
            self._write_readable()
        return self.directory / "memory.md", self.directory / "memory.json"

    def close(self):
        if self._readable_writer:
            self._readable_writer.close()
            self._readable_writer = None

    def observe(self, message: Observation) -> bool:
        now = self.clock()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO scopes(namespace,scope) VALUES(?,?)", (message.namespace, message.scope))
            row = db.execute("SELECT * FROM people WHERE namespace=? AND scope=? AND person_id=?",
                             (message.namespace, message.scope, message.sender_id)).fetchone()
            if row and row["paused"]:
                return False
            # Check before updating aliases: a replay cannot rewrite the current name.
            if db.execute("SELECT 1 FROM messages WHERE namespace=? AND scope=? AND message_id=?",
                          (message.namespace, message.scope, message.message_id)).fetchone():
                return False
            name = message.sender_name or (row["name"] if row else message.sender_id)
            aliases = json.loads(row["aliases"]) if row else []
            if name not in aliases:
                aliases = (aliases + [name])[-20:]
            visible_change = row is None or row["name"] != name or json.loads(row["aliases"]) != aliases
            db.execute("""INSERT INTO people VALUES(?,?,?,?,?,0,?)
                ON CONFLICT(namespace,scope,person_id) DO UPDATE SET
                name=excluded.name,aliases=excluded.aliases,updated_at=excluded.updated_at""",
                       (message.namespace, message.scope, message.sender_id, name, json.dumps(aliases, ensure_ascii=False), now))
            db.execute("INSERT INTO messages(namespace,scope,message_id,sender_id,sender_name,text,provider_id,timestamp,received_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (message.namespace, message.scope, message.message_id, message.sender_id, name, message.text,
                        message.provider_id, message.timestamp, now))
        if visible_change:
            self._sync_readable(None)
        return True

    def next_batch(self, *, delay: float, limit: int, mode="legacy", min_batch=1, max_wait=1800) -> dict | None:
        now = self.clock()
        with self.connection() as db:
            if mode == "person":
                row = db.execute("""SELECT namespace,scope,sender_id,COUNT(*) AS n,MIN(received_at) AS first,
                    MAX(received_at) AS last FROM messages WHERE done=0 AND retry_at<=?
                    GROUP BY namespace,scope,sender_id
                    HAVING n>=? OR (n>=? AND last<=?) OR first<=?
                    OR MAX(text LIKE '记住%' OR text LIKE '更正%' OR text LIKE '纠正%' OR text LIKE '以后%')
                    ORDER BY first LIMIT 1""", (now, limit, min_batch, now - delay, now - max_wait)).fetchone()
            else:
                row = db.execute("""SELECT namespace,scope,COUNT(*) AS n,MIN(received_at) AS first,
                    MAX(received_at) AS last FROM messages WHERE done=0 AND retry_at<=?
                    GROUP BY namespace,scope HAVING last<=? OR n>=? OR first<=?
                    ORDER BY first LIMIT 1""", (now, now - delay, limit, now - max(delay * 3, 30))).fetchone()
            if not row:
                return None
            ns, scope = row["namespace"], row["scope"]
            target_id = row["sender_id"] if mode == "person" else None
            target_sql = " AND sender_id=?" if target_id else ""
            params = [ns, scope, now] + ([target_id] if target_id else []) + [limit]
            messages = [dict(r) for r in db.execute("""SELECT message_id,sender_id,sender_name,text,provider_id,timestamp
                FROM messages WHERE namespace=? AND scope=? AND done=0 AND retry_at<=?"""
                + target_sql + " ORDER BY received_at,rowid LIMIT ?", params)]
            epoch = db.execute("SELECT epoch FROM scopes WHERE namespace=? AND scope=?", (ns, scope)).fetchone()[0]
            # Include old subjects/keys so corrections can update stable fields.
            if target_id:
                existing = [self.decode(r) for r in db.execute("""SELECT * FROM memory_details WHERE namespace=? AND scope=?
                    AND (expires_at IS NULL OR expires_at>?) AND subject_id IN (?, '@group')
                    ORDER BY (subject_id=?) DESC,core DESC,importance DESC,updated_at DESC LIMIT 120""", (ns, scope, now, target_id, target_id))]
            else:
                existing = [self.decode(r) for r in db.execute("""SELECT * FROM memory_details WHERE namespace=? AND scope=?
                    AND (expires_at IS NULL OR expires_at>?) ORDER BY core DESC,importance DESC,updated_at DESC LIMIT 60""", (ns, scope, now))]
            recent_context = []
            if target_id:
                claimed = {message["message_id"] for message in messages}
                budget = 12000
                for source in db.execute("""SELECT message_id,sender_id,sender_name,text,provider_id,timestamp
                    FROM messages m WHERE namespace=? AND scope=? AND NOT EXISTS
                    (SELECT 1 FROM people p WHERE p.namespace=m.namespace AND p.scope=m.scope
                     AND p.person_id=m.sender_id AND p.paused=1) ORDER BY received_at DESC,rowid DESC LIMIT 40""", (ns, scope)):
                    if source["message_id"] in claimed:
                        continue
                    item = dict(source)
                    item["text"] = item["text"][:1200]
                    if len(item["text"]) > budget:
                        continue
                    recent_context.append(item)
                    budget -= len(item["text"])
                recent_context.reverse()
            return {"namespace": ns, "scope": scope, "epoch": epoch, "target_id": target_id,
                    "messages": messages, "recent_context": recent_context, "existing": existing}

    @staticmethod
    def decode(row) -> dict:
        result = dict(row)
        for key in ("evidence_ids", "participant_ids", "evidence_days"):
            if key in result:
                result[key] = json.loads(result[key])
        return result

    def _candidate(self, item, batch, now):
        if not isinstance(item, dict):
            raise ValueError("Candidate must be an object")
        kind, subject, key, content = (item.get(k) for k in ("kind", "subject_id", "key", "content"))
        if kind not in KINDS or not isinstance(key, str) or not re.fullmatch(r"[\w.:-]{1,100}", key):
            raise ValueError("Invalid memory kind/key")
        if not isinstance(content, str) or not content.strip() or len(content) > 500:
            raise ValueError("Invalid memory content")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.8 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        evidence = item.get("evidence_ids")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 20 or not all(isinstance(e, str) for e in evidence):
            raise ValueError("Missing evidence")
        sources = {m["message_id"]: m for m in batch.get("recent_context", []) + batch["messages"]}
        if any(e not in sources for e in evidence):
            raise ValueError("Evidence is not in the observed batch")
        evidence = list(dict.fromkeys(evidence))
        if not set(evidence).intersection(m["message_id"] for m in batch["messages"]):
            raise ValueError("A memory needs evidence from the batch being processed")
        participants = item.get("participant_ids", [])
        if not isinstance(participants, list) or len(participants) > 30 or not all(isinstance(p, str) for p in participants):
            raise ValueError("Invalid participants")
        senders = {m["sender_id"] for m in sources.values()}
        if kind in GROUP_KINDS:
            if subject != "@group" or any(p not in senders for p in participants):
                raise ValueError("Invalid group memory attribution")
            participants = sorted(set(participants) | {sources[e]["sender_id"] for e in evidence})
            if kind == "group_profile" and len({sources[e]["sender_id"] for e in evidence}) < 2:
                raise ValueError("A group profile needs evidence from two speakers")
        elif subject not in senders or any(sources[e]["sender_id"] != subject for e in evidence) or participants:
            raise ValueError("A personal fact must be evidenced by that person's own message")
        elif batch.get("target_id") and subject != batch["target_id"]:
            raise ValueError("A person batch can only update its target profile")
        if kind == "style" and len(evidence) < 3:
            raise ValueError("An interaction observation needs three distinct messages")
        ttl = item.get("ttl_days")
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3650):
            raise ValueError("Invalid expiry")
        if kind in {"state", "style", "group_profile"}:
            ttl = min(ttl or 7, 30)
        elif kind in GROUP_KINDS:
            ttl = min(ttl or (60 if kind == "joke" else 90), 365)
        importance = item.get("importance", 3)
        if isinstance(importance, bool) or not isinstance(importance, int) or not 1 <= importance <= 5:
            raise ValueError("Invalid importance")
        retention = item.get("retention", "normal")
        if retention not in {"normal", "core"}:
            raise ValueError("Invalid retention")
        explicit_core = retention == "core" and any(re.search(r"永久记住|永远记住|一直记得|长期记住", sources[e]["text"])
                            and not re.search(r"(?:不要|不想|不用|不必|别).{0,4}(?:永久|永远|一直|长期)", sources[e]["text"])
                            and sources[e]["sender_id"] == subject
                            and e in {m["message_id"] for m in batch["messages"]} for e in evidence)
        return {"kind": kind, "subject_id": subject, "key": key, "content": content.strip(),
                "confidence": confidence, "evidence_ids": evidence, "participant_ids": participants,
                "expires_at": now + ttl * 86400 if ttl else None, "update_id": item.get("update_id"),
                "importance": importance, "retention": retention,
                "explicit_core": explicit_core}

    def _match_existing(self, db, namespace, scope, item):
        rows = list(db.execute("SELECT * FROM memories WHERE namespace=? AND scope=? AND subject_id=? AND kind=?",
                               (namespace, scope, item["subject_id"], item["kind"])))
        if item["update_id"]:
            if not isinstance(item["update_id"], str):
                raise ValueError("Invalid update_id")
            match = next((row for row in rows if row["id"] == item["update_id"]), None)
            if match is None:
                raise ValueError("update_id does not belong to this subject, scope and type")
            return match
        for row in rows:
            if normalized_key(row["key"], row["kind"]) == normalized_key(item["key"], item["kind"]):
                return row
        for row in rows:
            if same_content(row["content"], item["content"]):
                return row
        return None

    def _blocked(self, db, namespace, scope, item):
        content_key = "@content:" + hashlib.sha256(normalized_content(item["content"]).encode()).hexdigest()
        for row in db.execute("SELECT kind,key FROM blocks WHERE namespace=? AND scope=? AND subject_id=?",
                              (namespace, scope, item["subject_id"])):
            if row["kind"] == "*" and row["key"] == content_key:
                return True
            if row["kind"] == item["kind"] and normalized_key(row["key"], row["kind"]) == normalized_key(item["key"], item["kind"]):
                return True
        return False

    def _cap_memories(self, db, namespace, scope):
        subjects = [r[0] for r in db.execute("SELECT DISTINCT subject_id FROM memories WHERE namespace=? AND scope=?", (namespace, scope))]
        for subject in subjects:
            cap = self.max_group_memories if subject == "@group" else self.max_person_memories
            rows = db.execute("""SELECT id FROM memory_details WHERE namespace=? AND scope=? AND subject_id=?
                AND kind!='boundary' AND core=0
                ORDER BY importance DESC,(kind='group_profile') DESC,json_array_length(evidence_days) DESC,
                updated_at DESC,created_at DESC,id DESC LIMIT -1 OFFSET ?""", (namespace, scope, subject, cap))
            for row in list(rows):
                db.execute("DELETE FROM memories WHERE id=?", (row["id"],))
                db.execute("DELETE FROM changes WHERE memory_id=?", (row["id"],))

    def apply_batch(self, batch: dict, candidates: list[dict]) -> tuple[int, int]:
        now = self.clock()
        written = rejected = unchanged = 0
        ns, scope = batch["namespace"], batch["scope"]
        with self.connection(sync_readable=True) as db:
            epoch = db.execute("SELECT epoch FROM scopes WHERE namespace=? AND scope=?", (ns, scope)).fetchone()
            if not epoch or epoch[0] != batch["epoch"]:
                return 0, len(candidates)  # Forget/pause invalidates in-flight model results.
            for candidate in candidates[:12]:
                try:
                    item = self._candidate(candidate, batch, now)
                    old = self._match_existing(db, ns, scope, item)
                except (ValueError, TypeError):
                    rejected += 1
                    continue
                if old:
                    item["key"] = old["key"]
                if self._blocked(db, ns, scope, item):
                    rejected += 1
                    continue
                if old and not set(item["evidence_ids"]).difference(json.loads(old["evidence_ids"])):
                    # Re-interpreting the same sources must not rewrite a profile or renew its TTL.
                    unchanged += 1
                    continue
                source_ids = list(item["evidence_ids"])
                memory_id = "m_" + uuid.uuid4().hex[:16]
                if old:
                    memory_id = old["id"]
                    # A correction drops old contradictory evidence; a duplicate reinforces it.
                    if same_content(old["content"], item["content"]):
                        item["evidence_ids"] = list(dict.fromkeys(json.loads(old["evidence_ids"]) + item["evidence_ids"]))[-20:]
                        item["participant_ids"] = sorted(set(json.loads(old["participant_ids"])) | set(item["participant_ids"]))
                policy = db.execute("SELECT * FROM memory_policy WHERE memory_id=?", (memory_id,)).fetchone()
                # Only newly received, newly cited batch evidence can reinforce permanence.
                fresh_ids = set(source_ids).intersection(m["message_id"] for m in batch["messages"])
                if old:
                    fresh_ids.difference_update(json.loads(old["evidence_ids"]))
                days = {datetime.fromtimestamp(row[0], timezone.utc).date().isoformat() for mid in fresh_ids
                        for row in db.execute("SELECT received_at FROM messages WHERE namespace=? AND scope=? AND message_id=?",
                                              (ns, scope, mid))}
                same_topic_content = old is not None and same_content(old["content"], item["content"])
                if policy and same_topic_content:
                    days.update(json.loads(policy["evidence_days"]))
                days = sorted(days)[-30:]
                pinned = policy["pinned"] if policy else 0
                core = bool(policy and policy["core"])
                importance = max(item["importance"], policy["importance"] if policy and same_topic_content else 1)
                automatic = (item["retention"] == "core" and importance == 5 and len(days) >= 3
                             and pinned != -1 and (item["subject_id"] != "@group" or len(item["participant_ids"]) >= 2))
                core_count = db.execute("""SELECT COUNT(*) FROM memory_details
                    WHERE namespace=? AND scope=? AND subject_id=? AND core=1""", (ns, scope, item["subject_id"])).fetchone()[0]
                if item["kind"] in CORE_KINDS and (item["explicit_core"] or (automatic and core_count < 30)):
                    core = True
                if core:
                    item["expires_at"] = None
                db.execute("""INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    ON CONFLICT(namespace,scope,subject_id,kind,key) DO UPDATE SET
                    content=excluded.content,evidence_ids=excluded.evidence_ids,participant_ids=excluded.participant_ids,
                    confidence=excluded.confidence,updated_at=excluded.updated_at,expires_at=excluded.expires_at""",
                           (memory_id, ns, scope, item["subject_id"], item["kind"], item["key"], item["content"],
                            json.dumps(item["evidence_ids"]), json.dumps(item["participant_ids"]), item["confidence"],
                            now, now, item["expires_at"]))
                db.execute("INSERT OR REPLACE INTO memory_policy VALUES(?,?,?,?,?)",
                           (memory_id, importance, int(core), pinned, json.dumps(days)))
                db.execute("INSERT INTO changes(memory_id,action,timestamp) VALUES(?,?,?)", (memory_id, "update" if old else "add", now))
                written += 1
            self._cap_memories(db, ns, scope)
            for message in batch["messages"]:
                db.execute("UPDATE messages SET done=1 WHERE namespace=? AND scope=? AND message_id=?", (ns, scope, message["message_id"]))
            outcome = "saved" if written else ("no_candidates" if not candidates else
                                                "unchanged" if unchanged else "validation_rejected")
            db.execute("INSERT INTO summary_runs(namespace,scope,subject_id,finished_at,messages,written,rejected,outcome,error_type) VALUES(?,?,?,?,?,?,?,?,?)",
                       (ns, scope, batch.get("target_id"), now, len(batch["messages"]), written, rejected, outcome, ""))
        return written, rejected

    def fail_batch(self, batch, error_type=""):
        with self.connection() as db:
            epoch = db.execute("SELECT epoch FROM scopes WHERE namespace=? AND scope=?",
                               (batch["namespace"], batch["scope"])).fetchone()
            if not epoch or epoch[0] != batch["epoch"]:
                return
            for message in batch["messages"]:
                row = db.execute("SELECT attempts FROM messages WHERE namespace=? AND scope=? AND message_id=? AND done=0",
                                 (batch["namespace"], batch["scope"], message["message_id"])).fetchone()
                if row:
                    delay = min(21600, 30 * 2 ** min(row["attempts"], 10))
                    db.execute("""UPDATE messages SET attempts=attempts+1,retry_at=?
                        WHERE namespace=? AND scope=? AND message_id=? AND done=0""",
                               (self.clock() + delay, batch["namespace"], batch["scope"], message["message_id"]))
            db.execute("INSERT INTO summary_runs(namespace,scope,subject_id,finished_at,messages,written,rejected,outcome,error_type) VALUES(?,?,?,?,?,0,0,'error',?)",
                       (batch["namespace"], batch["scope"], batch.get("target_id"), self.clock(), len(batch["messages"]), error_type[:80]))

    def context(self, namespace, scope, sender_id, query, *, limit=8, char_budget=5000, cooldown=21600):
        now = self.clock()
        limit = max(1, min(30, int(limit)))
        char_budget = max(0, min(20000, int(char_budget)))
        query_terms = terms(query)
        with self.connection() as db:
            people = list(db.execute("SELECT person_id,name,aliases FROM people WHERE namespace=? AND scope=?", (namespace, scope)))
            names = {r["person_id"]: r["name"] for r in people}
            person_terms = {r["person_id"]: terms(" ".join(json.loads(r["aliases"]))) for r in people}
            mentioned = {r["person_id"] for r in people if any(len(alias) >= 2 and alias.casefold() in query.casefold()
                         for alias in json.loads(r["aliases"]))}
            # Rank within the current scope before truncating the candidate pool.
            # This keeps an old matching memory reachable after a group exceeds 1000 rows.
            ranking = ["CASE WHEN subject_id=? THEN 5 ELSE 0 END", "CASE WHEN kind='group_profile' THEN 8 ELSE 0 END",
                       "core*2", "importance-3"]
            ranking_params = [sender_id]
            if mentioned:
                ranking.append("CASE WHEN subject_id IN (" + ",".join("?" for _ in mentioned) + ") THEN 8 ELSE 0 END")
                ranking_params.extend(sorted(mentioned))
            for term in sorted(query_terms, key=lambda t: (-len(t), t))[:24]:
                ranking.append("CASE WHEN instr(lower(key || ' ' || content),?)>0 THEN 3 ELSE 0 END")
                ranking_params.append(term)
            rows = db.execute("""SELECT * FROM memory_details WHERE namespace=? AND scope=?
                AND (expires_at IS NULL OR expires_at>?) ORDER BY (kind='boundary') DESC,("""
                + "+".join(ranking) + ") DESC,updated_at DESC LIMIT 1000",
                [namespace, scope, now, *ranking_params])
            scored = []
            for row in rows:
                item = self.decode(row)
                overlap = len(query_terms & terms(item["key"] + " " + item["content"]))
                overlap += 2 * len(query_terms & person_terms.get(item["subject_id"], set()))
                own = item["subject_id"] == sender_id
                # Boundaries remain visible, but unrelated other-person information does not.
                if not own and item["kind"] not in {"boundary", "group_profile"} and not overlap:
                    continue
                if item["kind"] == "joke" and item["last_used"] and now - item["last_used"] < cooldown:
                    continue
                score = overlap * 3 + (5 if own else 0) + (8 if item["subject_id"] in mentioned else 0) + (100 if item["kind"] == "boundary" else 0)
                if item["kind"] == "group_profile":
                    score += 8
                score += item["core"] * 2 + item["importance"] - 3
                if own and item["kind"] in {"fact", "state"} and not overlap:
                    score -= 3
                scored.append((score, item["updated_at"], item))
            selected = []
            used = 0
            group_profiles = 0
            for _, _, item in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True):
                if item["kind"] == "group_profile" and group_profiles >= 2:
                    continue
                public = {k: item[k] for k in ("id", "subject_id", "kind", "content", "updated_at", "participant_ids")}
                cost = len(json.dumps(public, ensure_ascii=False))
                if used + cost > char_budget:
                    continue
                selected.append(public)
                group_profiles += item["kind"] == "group_profile"
                used += cost
                if item["kind"] == "joke":
                    db.execute("UPDATE memories SET last_used=? WHERE id=?", (now, item["id"]))
                if len(selected) >= limit:
                    break
            referenced = {sender_id}
            for item in selected:
                referenced.add(item["subject_id"])
                referenced.update(item["participant_ids"])
            return {"current_sender_id": sender_id, "person_names": {p: names[p] for p in referenced if p in names}, "memories": selected}

    def profile(self, namespace, scope, person_id):
        now = self.clock()
        with self.connection() as db:
            person = db.execute("SELECT * FROM people WHERE namespace=? AND scope=? AND person_id=?", (namespace, scope, person_id)).fetchone()
            memories = [self.decode(r) for r in db.execute("""SELECT * FROM memory_details WHERE namespace=? AND scope=?
                AND subject_id=? AND (expires_at IS NULL OR expires_at>?) ORDER BY core DESC,importance DESC,updated_at DESC""", (namespace, scope, person_id, now))]
            row = db.execute("""SELECT COUNT(*) AS retained,
                COALESCE(SUM(done=1),0) AS processed,
                COALESCE(SUM(done=0),0) AS pending,
                COALESCE(SUM(done=0 AND attempts>=3),0) AS failed,
                COALESCE(SUM(done=0 AND attempts>0),0) AS retrying,
                MIN(CASE WHEN done=0 THEN received_at END) AS oldest_pending_at,
                MIN(CASE WHEN done=0 AND attempts>0 THEN retry_at END) AS next_retry_at
                FROM messages WHERE namespace=? AND scope=? AND sender_id=?""", (namespace, scope, person_id)).fetchone()
            stats = dict(row)
            stats["group_memories"] = db.execute("""SELECT COUNT(*) FROM memories WHERE namespace=? AND scope=?
                AND subject_id='@group' AND (expires_at IS NULL OR expires_at>?)""", (namespace, scope, now)).fetchone()[0]
            last = db.execute("""SELECT finished_at,messages,written,rejected,outcome,error_type FROM summary_runs
                WHERE namespace=? AND scope=? AND subject_id=? ORDER BY seq DESC LIMIT 1""", (namespace, scope, person_id)).fetchone()
            stats["last_summary"] = dict(last) if last else None
            stats["personal_content_chars"] = sum(len(memory["content"]) for memory in memories)
            stats["core_memories"] = sum(bool(memory["core"]) for memory in memories)
            return {"person_id": person_id, "name": person["name"] if person else "", "paused": bool(person["paused"]) if person else False,
                    "aliases": json.loads(person["aliases"]) if person else [], "memories": memories, "stats": stats}

    def _invalidate(self, db, namespace, scope, person_id=None):
        db.execute("INSERT OR IGNORE INTO scopes(namespace,scope) VALUES(?,?)", (namespace, scope))
        db.execute("UPDATE scopes SET epoch=epoch+1 WHERE namespace=? AND scope=?", (namespace, scope))
        # Reject all in-flight results, but leave unrelated members' queues intact.
        if person_id is not None:
            db.execute("UPDATE messages SET done=1 WHERE namespace=? AND scope=? AND sender_id=?",
                       (namespace, scope, person_id))

    def pin(self, namespace, scope, requester_id, memory_id, *, permanent=True, admin=False):
        with self.connection(sync_readable=True) as db:
            item = db.execute("SELECT * FROM memory_details WHERE namespace=? AND scope=? AND id=?",
                              (namespace, scope, memory_id)).fetchone()
            if not item or (item["expires_at"] is not None and item["expires_at"] <= self.clock()):
                return False
            if item["subject_id"] != requester_id and not admin:
                raise PermissionError("Only the subject or an administrator may pin a memory")
            if item["kind"] not in CORE_KINDS:
                raise ValueError("Temporary states, observations and boundaries cannot be pinned")
            self._invalidate(db, namespace, scope)
            ttl = 60 if item["kind"] == "joke" else 90 if item["kind"] == "event" else None
            expires_at = None if permanent or ttl is None else self.clock() + ttl * 86400
            db.execute("UPDATE memories SET expires_at=? WHERE id=?", (expires_at, memory_id))
            db.execute("INSERT OR REPLACE INTO memory_policy VALUES(?,?,?,?,?)",
                       (memory_id, max(item["importance"], 5) if permanent else item["importance"],
                        int(permanent), 1 if permanent else -1, item["evidence_days"]))
            db.execute("INSERT INTO changes(memory_id,action,timestamp) VALUES(?,?,?)",
                       (memory_id, "pin" if permanent else "unpin", self.clock()))
            return True

    def forget(self, namespace, scope, requester_id, memory_id, *, admin=False):
        with self.connection(sync_readable=True) as db:
            row = db.execute("SELECT * FROM memories WHERE namespace=? AND scope=? AND id=?", (namespace, scope, memory_id)).fetchone()
            if not row:
                return False
            item = self.decode(row)
            if item["kind"] == "group_profile" and not admin:
                raise PermissionError("Group-wide profiles are managed by administrators")
            if not admin and requester_id != item["subject_id"] and requester_id not in item["participant_ids"]:
                raise PermissionError("Only the subject, a participant, or an administrator may forget this memory")
            self._invalidate(db, namespace, scope, item["subject_id"] if item["subject_id"] != "@group" else None)
            db.execute("INSERT OR IGNORE INTO blocks VALUES(?,?,?,?,?)", (namespace, scope, item["subject_id"], item["kind"], item["key"]))
            content_key = "@content:" + hashlib.sha256(normalized_content(item["content"]).encode()).hexdigest()
            db.execute("INSERT OR IGNORE INTO blocks VALUES(?,?,?,?,?)", (namespace, scope, item["subject_id"], "*", content_key))
            for source_id in item["evidence_ids"]:
                db.execute("DELETE FROM messages WHERE namespace=? AND scope=? AND message_id=?", (namespace, scope, source_id))
            db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            db.execute("DELETE FROM changes WHERE memory_id=?", (memory_id,))
            return True

    def pause(self, namespace, scope, person_id, *, erase=False):
        with self.connection(sync_readable=True) as db:
            self._invalidate(db, namespace, scope, person_id)
            db.execute("""INSERT INTO people VALUES(?,?,?,'','[]',1,?) ON CONFLICT(namespace,scope,person_id)
                DO UPDATE SET paused=1""", (namespace, scope, person_id, self.clock()))
            if erase:
                ids = set()
                evidence = {r[0] for r in db.execute("SELECT message_id FROM messages WHERE namespace=? AND scope=? AND sender_id=?", (namespace, scope, person_id))}
                for row in db.execute("SELECT * FROM memories WHERE namespace=? AND scope=?", (namespace, scope)):
                    item = self.decode(row)
                    if item["subject_id"] == person_id or person_id in item["participant_ids"] or evidence.intersection(item["evidence_ids"]):
                        ids.add(item["id"])
                for mid in ids:
                    db.execute("DELETE FROM memories WHERE id=?", (mid,))
                    db.execute("DELETE FROM changes WHERE memory_id=?", (mid,))
                db.execute("DELETE FROM messages WHERE namespace=? AND scope=? AND sender_id=?", (namespace, scope, person_id))
                db.execute("UPDATE people SET name='',aliases='[]' WHERE namespace=? AND scope=? AND person_id=?", (namespace, scope, person_id))
                db.execute("DELETE FROM summary_runs WHERE namespace=? AND scope=? AND subject_id=?", (namespace, scope, person_id))

    def resume(self, namespace, scope, person_id):
        with self.connection(sync_readable=True) as db:
            db.execute("UPDATE people SET paused=0 WHERE namespace=? AND scope=? AND person_id=?", (namespace, scope, person_id))

    def retry_pending(self, namespace, scope, person_id):
        with self.connection() as db:
            return db.execute("""UPDATE messages SET attempts=0,retry_at=0
                WHERE namespace=? AND scope=? AND sender_id=? AND done=0 AND attempts>0""",
                (namespace, scope, person_id)).rowcount

    def prune(self, retention_days=14):
        with self.connection(sync_readable=True) as db:
            # Never delete a pending/retry job before its extraction has settled.
            db.execute("DELETE FROM messages WHERE done=1 AND received_at<?", (self.clock() - retention_days * 86400,))
            db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at<=?", (self.clock(),))
            db.execute("DELETE FROM summary_runs WHERE finished_at<?", (self.clock() - retention_days * 86400,))

    def backup(self) -> Path:
        destination = self.directory / ("backup-" + str(time.time_ns()) + ".sqlite3")
        with self.connection() as db:
            backup = sqlite3.connect(destination)
            try:
                db.backup(backup)
            finally:
                backup.close()
        return destination

    def automatic_backup(self, keep_days=7) -> Path:
        """Keep a daily SQLite snapshot and policy; never rotate manual backups."""
        keep_days = max(1, min(90, int(keep_days)))
        today = datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()
        directory = self.directory / "backups"
        directory.mkdir(exist_ok=True)
        destination = directory / ("auto-" + today + ".sqlite3")
        if not destination.exists():
            temporary = directory / (".cdmem-backup-" + uuid.uuid4().hex + ".tmp")
            try:
                # A separate WAL reader avoids holding the message-path Python lock.
                reader = sqlite3.connect(self.path, timeout=10)
                try:
                    reader.execute("PRAGMA query_only=ON")
                    reader.execute("BEGIN")
                    backup = sqlite3.connect(temporary)
                    try:
                        reader.backup(backup, pages=256)
                    finally:
                        backup.close()
                finally:
                    reader.close()
                # Write the policy before committing the snapshot so a failed policy
                # copy is retried rather than silently producing an incomplete pair.
                atomic_write(destination.with_suffix(".summary_prompt.txt"),
                             self.prompt_path.read_text(encoding="utf-8-sig") if self.prompt_path.exists() else DEFAULT_SUMMARY_POLICY)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        oldest = datetime.fromtimestamp(self.clock() - (keep_days - 1) * 86400, timezone.utc).date().isoformat()
        for previous in directory.glob("auto-*.sqlite3"):
            if re.fullmatch(r"auto-\d{4}-\d{2}-\d{2}\.sqlite3", previous.name) and previous.name[5:15] < oldest:
                try:
                    previous.unlink()
                    previous.with_suffix(".summary_prompt.txt").unlink(missing_ok=True)
                except OSError as exc:
                    # An administrator may have an old snapshot open on Windows.
                    logger.warning("Chuanda automatic backup rotation deferred (%s)", type(exc).__name__)
        return destination
