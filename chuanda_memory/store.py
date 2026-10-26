from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import Observation
from .readable import CoalescingWriter, snapshot, write_views

KINDS = {"fact", "preference", "boundary", "state", "event", "joke"}
GROUP_KINDS = {"event", "joke"}
logger = logging.getLogger(__name__)


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

    def __init__(self, data_dir: str | Path, *, clock=time.time, readable_background=False):
        self.directory = Path(data_dir).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "memory.sqlite3"
        self.clock = clock
        self.lock = threading.RLock()
        self.readable_background = readable_background
        self._readable_writer = None
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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, namespace TEXT, scope TEXT, subject_id TEXT,
                    kind TEXT, key TEXT, content TEXT, evidence_ids TEXT,
                    participant_ids TEXT, confidence REAL, created_at REAL,
                    updated_at REAL, expires_at REAL, last_used REAL NOT NULL DEFAULT 0,
                    UNIQUE(namespace, scope, subject_id, kind, key));
                CREATE INDEX IF NOT EXISTS memories_scope ON memories(namespace, scope, subject_id);
                CREATE TABLE IF NOT EXISTS blocks (
                    namespace TEXT, scope TEXT, subject_id TEXT, kind TEXT, key TEXT,
                    PRIMARY KEY(namespace, scope, subject_id, kind, key));
                CREATE TABLE IF NOT EXISTS changes (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT,
                    action TEXT, timestamp REAL);
                PRAGMA user_version=1;
            """)

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
        with self.connection(sync_readable=True) as db:
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
            db.execute("""INSERT INTO people VALUES(?,?,?,?,?,0,?)
                ON CONFLICT(namespace,scope,person_id) DO UPDATE SET
                name=excluded.name,aliases=excluded.aliases,updated_at=excluded.updated_at""",
                       (message.namespace, message.scope, message.sender_id, name, json.dumps(aliases, ensure_ascii=False), now))
            db.execute("INSERT INTO messages(namespace,scope,message_id,sender_id,sender_name,text,provider_id,timestamp,received_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (message.namespace, message.scope, message.message_id, message.sender_id, name, message.text,
                        message.provider_id, message.timestamp, now))
            return True

    def next_batch(self, *, delay: float, limit: int) -> dict | None:
        now = self.clock()
        with self.connection() as db:
            row = db.execute("""SELECT namespace,scope,COUNT(*) AS n,MIN(received_at) AS first,
                MAX(received_at) AS last FROM messages WHERE done=0 AND retry_at<=? AND attempts<3
                GROUP BY namespace,scope HAVING last<=? OR n>=? OR first<=?
                ORDER BY first LIMIT 1""", (now, now - delay, limit, now - max(delay * 3, 30))).fetchone()
            if not row:
                return None
            ns, scope = row["namespace"], row["scope"]
            messages = [dict(r) for r in db.execute("""SELECT message_id,sender_id,sender_name,text,provider_id,timestamp
                FROM messages WHERE namespace=? AND scope=? AND done=0 AND retry_at<=? AND attempts<3
                ORDER BY received_at,rowid LIMIT ?""", (ns, scope, now, limit))]
            epoch = db.execute("SELECT epoch FROM scopes WHERE namespace=? AND scope=?", (ns, scope)).fetchone()[0]
            # Include old subjects/keys so corrections can update stable fields.
            existing = [self.decode(r) for r in db.execute("""SELECT * FROM memories WHERE namespace=? AND scope=?
                AND (expires_at IS NULL OR expires_at>?) ORDER BY updated_at DESC LIMIT 60""", (ns, scope, now))]
            return {"namespace": ns, "scope": scope, "epoch": epoch, "messages": messages, "existing": existing}

    @staticmethod
    def decode(row) -> dict:
        result = dict(row)
        for key in ("evidence_ids", "participant_ids"):
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
        sources = {m["message_id"]: m for m in batch["messages"]}
        if any(e not in sources for e in evidence):
            raise ValueError("Evidence is not in the observed batch")
        evidence = list(dict.fromkeys(evidence))
        participants = item.get("participant_ids", [])
        if not isinstance(participants, list) or len(participants) > 30 or not all(isinstance(p, str) for p in participants):
            raise ValueError("Invalid participants")
        senders = {m["sender_id"] for m in batch["messages"]}
        if kind in GROUP_KINDS:
            if subject != "@group" or any(p not in senders for p in participants):
                raise ValueError("Invalid group memory attribution")
            participants = sorted(set(participants) | {sources[e]["sender_id"] for e in evidence})
        elif subject not in senders or any(sources[e]["sender_id"] != subject for e in evidence) or participants:
            raise ValueError("A personal fact must be evidenced by that person's own message")
        ttl = item.get("ttl_days")
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3650):
            raise ValueError("Invalid expiry")
        if kind == "state":
            ttl = min(ttl or 7, 30)
        elif kind in GROUP_KINDS:
            ttl = min(ttl or (60 if kind == "joke" else 90), 365)
        return {"kind": kind, "subject_id": subject, "key": key, "content": content.strip(),
                "confidence": confidence, "evidence_ids": evidence, "participant_ids": participants,
                "expires_at": now + ttl * 86400 if ttl else None}

    def apply_batch(self, batch: dict, candidates: list[dict]) -> tuple[int, int]:
        now = self.clock()
        written = rejected = 0
        ns, scope = batch["namespace"], batch["scope"]
        with self.connection(sync_readable=True) as db:
            epoch = db.execute("SELECT epoch FROM scopes WHERE namespace=? AND scope=?", (ns, scope)).fetchone()
            if not epoch or epoch[0] != batch["epoch"]:
                return 0, len(candidates)  # Forget/pause invalidates in-flight model results.
            for candidate in candidates[:12]:
                try:
                    item = self._candidate(candidate, batch, now)
                except (ValueError, TypeError):
                    rejected += 1
                    continue
                if db.execute("SELECT 1 FROM blocks WHERE namespace=? AND scope=? AND subject_id=? AND kind=? AND key=?",
                              (ns, scope, item["subject_id"], item["kind"], item["key"])).fetchone():
                    rejected += 1
                    continue
                memory_id = "m_" + uuid.uuid4().hex[:16]
                old = db.execute("SELECT id FROM memories WHERE namespace=? AND scope=? AND subject_id=? AND kind=? AND key=?",
                                 (ns, scope, item["subject_id"], item["kind"], item["key"])).fetchone()
                if old:
                    memory_id = old[0]
                db.execute("""INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    ON CONFLICT(namespace,scope,subject_id,kind,key) DO UPDATE SET
                    content=excluded.content,evidence_ids=excluded.evidence_ids,participant_ids=excluded.participant_ids,
                    confidence=excluded.confidence,updated_at=excluded.updated_at,expires_at=excluded.expires_at""",
                           (memory_id, ns, scope, item["subject_id"], item["kind"], item["key"], item["content"],
                            json.dumps(item["evidence_ids"]), json.dumps(item["participant_ids"]), item["confidence"],
                            now, now, item["expires_at"]))
                db.execute("INSERT INTO changes(memory_id,action,timestamp) VALUES(?,?,?)", (memory_id, "update" if old else "add", now))
                written += 1
            for message in batch["messages"]:
                db.execute("UPDATE messages SET done=1 WHERE namespace=? AND scope=? AND message_id=?", (ns, scope, message["message_id"]))
        return written, rejected

    def fail_batch(self, batch):
        with self.connection() as db:
            for message in batch["messages"]:
                db.execute("""UPDATE messages SET attempts=attempts+1,retry_at=? + 30 * (attempts+1)
                    WHERE namespace=? AND scope=? AND message_id=? AND done=0""",
                           (self.clock(), batch["namespace"], batch["scope"], message["message_id"]))

    def context(self, namespace, scope, sender_id, query, *, limit=8, char_budget=5000, cooldown=21600):
        now = self.clock()
        limit = max(1, min(30, int(limit)))
        char_budget = max(0, min(20000, int(char_budget)))
        query_terms = terms(query)
        with self.connection() as db:
            # Bounded candidate pool; first-version retrieval is deliberately lexical.
            rows = db.execute("""SELECT * FROM memories WHERE namespace=? AND scope=?
                AND (expires_at IS NULL OR expires_at>?)
                ORDER BY (kind='boundary') DESC,(subject_id=?) DESC,updated_at DESC LIMIT 1000""",
                              (namespace, scope, now, sender_id))
            people = list(db.execute("SELECT person_id,name,aliases FROM people WHERE namespace=? AND scope=?", (namespace, scope)))
            names = {r["person_id"]: r["name"] for r in people}
            person_terms = {r["person_id"]: terms(" ".join(json.loads(r["aliases"]))) for r in people}
            scored = []
            for row in rows:
                item = self.decode(row)
                overlap = len(query_terms & terms(item["key"] + " " + item["content"]))
                overlap += 2 * len(query_terms & person_terms.get(item["subject_id"], set()))
                own = item["subject_id"] == sender_id
                # Boundaries remain visible, but unrelated other-person information does not.
                if not own and item["kind"] != "boundary" and not overlap:
                    continue
                if item["kind"] == "joke" and item["last_used"] and now - item["last_used"] < cooldown:
                    continue
                score = overlap * 3 + (5 if own else 0) + (100 if item["kind"] == "boundary" else 0)
                if own and item["kind"] in {"fact", "state"} and not overlap:
                    score -= 3
                scored.append((score, item["updated_at"], item))
            selected = []
            used = 0
            for _, _, item in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True):
                public = {k: item[k] for k in ("id", "subject_id", "kind", "content", "updated_at", "participant_ids")}
                cost = len(json.dumps(public, ensure_ascii=False))
                if used + cost > char_budget:
                    continue
                selected.append(public)
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
            memories = [self.decode(r) for r in db.execute("""SELECT * FROM memories WHERE namespace=? AND scope=?
                AND subject_id=? AND (expires_at IS NULL OR expires_at>?) ORDER BY updated_at DESC""", (namespace, scope, person_id, now))]
            return {"person_id": person_id, "name": person["name"] if person else "", "paused": bool(person["paused"]) if person else False,
                    "aliases": json.loads(person["aliases"]) if person else [], "memories": memories}

    def _invalidate(self, db, namespace, scope):
        db.execute("INSERT OR IGNORE INTO scopes(namespace,scope) VALUES(?,?)", (namespace, scope))
        db.execute("UPDATE scopes SET epoch=epoch+1 WHERE namespace=? AND scope=?", (namespace, scope))
        # Old queued observations must not resurrect forgotten information.
        db.execute("UPDATE messages SET done=1 WHERE namespace=? AND scope=?", (namespace, scope))

    def forget(self, namespace, scope, requester_id, memory_id, *, admin=False):
        with self.connection(sync_readable=True) as db:
            row = db.execute("SELECT * FROM memories WHERE namespace=? AND scope=? AND id=?", (namespace, scope, memory_id)).fetchone()
            if not row:
                return False
            item = self.decode(row)
            if not admin and requester_id != item["subject_id"] and requester_id not in item["participant_ids"]:
                raise PermissionError("Only the subject, a participant, or an administrator may forget this memory")
            self._invalidate(db, namespace, scope)
            db.execute("INSERT OR IGNORE INTO blocks VALUES(?,?,?,?,?)", (namespace, scope, item["subject_id"], item["kind"], item["key"]))
            for source_id in item["evidence_ids"]:
                db.execute("DELETE FROM messages WHERE namespace=? AND scope=? AND message_id=?", (namespace, scope, source_id))
            db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            db.execute("DELETE FROM changes WHERE memory_id=?", (memory_id,))
            return True

    def pause(self, namespace, scope, person_id, *, erase=False):
        with self.connection(sync_readable=True) as db:
            self._invalidate(db, namespace, scope)
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

    def resume(self, namespace, scope, person_id):
        with self.connection(sync_readable=True) as db:
            db.execute("UPDATE people SET paused=0 WHERE namespace=? AND scope=? AND person_id=?", (namespace, scope, person_id))

    def prune(self, retention_days=14):
        with self.connection(sync_readable=True) as db:
            # Never delete a pending/retry job before its extraction has settled.
            db.execute("DELETE FROM messages WHERE (done=1 OR attempts>=3) AND received_at<?", (self.clock() - retention_days * 86400,))
            db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at<=?", (self.clock(),))

    def backup(self) -> Path:
        destination = self.directory / ("backup-" + str(time.time_ns()) + ".sqlite3")
        with self.connection() as db:
            backup = sqlite3.connect(destination)
            try:
                db.backup(backup)
            finally:
                backup.close()
        return destination
