"""Rebuildable UTF-8 views of committed memory, without raw chat transcripts."""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

LABELS = {"fact": "基本信息", "preference": "偏好", "boundary": "相处边界", "state": "近期状态",
          "style": "互动观察（有时效）", "event": "共同经历", "joke": "群内梗", "group_profile": "群整体记忆"}


class CoalescingWriter:
    """One writer; message handlers only signal an event and never wait for export."""
    def __init__(self, refresh, delay=0.5):
        self.refresh = refresh
        self.delay = delay
        self.changed = threading.Event()
        self.stopping = threading.Event()
        self.write_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="cdmem-readable", daemon=True)
        self.thread.start()

    def notify(self):
        self.changed.set()

    def flush(self):
        with self.write_lock:
            self.changed.clear()
            self.refresh()

    def _run(self):
        while True:
            self.changed.wait()
            self.stopping.wait(self.delay)
            self.flush()
            if self.stopping.is_set() and not self.changed.is_set():
                return

    def close(self):
        self.stopping.set()
        self.changed.set()
        self.thread.join()


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds") if value is not None else "长期"


def md(value):
    text = html.escape(str(value)).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#!|])", r"\\\1", text)


def snapshot(db, now):
    scopes = {}
    for row in db.execute("SELECT namespace,scope FROM scopes ORDER BY namespace,scope"):
        scopes[(row["namespace"], row["scope"])] = {"namespace": row["namespace"], "scope": row["scope"],
            "people": [], "group_memories": [], "blocked_topics": []}
    people = {}
    for row in db.execute("SELECT * FROM people ORDER BY namespace,scope,person_id"):
        person = {"person_id": row["person_id"], "name": row["name"], "aliases": json.loads(row["aliases"]),
                  "paused": bool(row["paused"]), "memories": []}
        scopes[(row["namespace"], row["scope"])]["people"].append(person)
        people[(row["namespace"], row["scope"], row["person_id"])] = person
    for row in db.execute("""SELECT * FROM memory_details WHERE expires_at IS NULL OR expires_at>?
        ORDER BY namespace,scope,subject_id,kind,key,id""", (now,)):
        memory = {key: row[key] for key in ("id", "subject_id", "kind", "key", "content", "confidence", "created_at", "updated_at", "expires_at")}
        memory["evidence_ids"] = json.loads(row["evidence_ids"])
        memory["participant_ids"] = json.loads(row["participant_ids"])
        memory["importance"] = row["importance"]
        memory["core"] = bool(row["core"])
        memory["pinned"] = row["pinned"] == 1
        scope = scopes[(row["namespace"], row["scope"])]
        if row["subject_id"] == "@group":
            scope["group_memories"].append(memory)
        else:
            people[(row["namespace"], row["scope"], row["subject_id"])]["memories"].append(memory)
    for row in db.execute("SELECT * FROM blocks WHERE kind!='*' ORDER BY namespace,scope,subject_id,kind,key"):
        scopes[(row["namespace"], row["scope"])]["blocked_topics"].append({key: row[key] for key in ("subject_id", "kind", "key")})
    return {"format_version": 1, "database_schema_version": 1, "generation": uuid.uuid4().hex,
            "generated_at": timestamp(now), "scopes": list(scopes.values())}


def markdown(data):
    lines = ["# Chuanda Memory", "", f"更新时间：{data['generated_at']}（UTC）", f"快照编号：{data['generation']}", "",
             "自动同步的本地查看副本；SQLite 是事实来源，手动修改此文件不会回写。", ""]
    def memories(items):
        if not items:
            lines.extend(["暂无有效记忆。", ""])
        for memory in items:
            lines.extend([f"- **{LABELS.get(memory['kind'], md(memory['kind']))}**：{md(memory['content'])}",
                          f"  - 保留策略：{'永久核心' if memory.get('core') else '普通'}；重要性：{memory.get('importance', 3)}/5",
                          f"  - ID：{md(memory['id'])}；主题：{md(memory['key'])}；置信度：{memory['confidence']:g}",
                          f"  - 更新时间：{timestamp(memory['updated_at'])}；有效期至：{timestamp(memory['expires_at'])}",
                          f"  - 来源消息 ID：{', '.join(md(e) for e in memory['evidence_ids'])}"])
            if memory["participant_ids"]:
                lines.append("  - 参与者 ID：" + ", ".join(md(p) for p in memory["participant_ids"]))
            lines.append("")
    if not data["scopes"]:
        lines.extend(["暂无记忆。", ""])
    for scope in data["scopes"]:
        lines.extend([f"## {md(scope['scope'])}", "", f"身份命名空间：{md(scope['namespace'])}", ""])
        for person in scope["people"]:
            lines.extend([f"### {md(person['name'] or person['person_id'])} · ID {md(person['person_id'])}", "",
                          f"采集状态：{'暂停' if person['paused'] else '开启'}",
                          "曾用名：" + ("、".join(md(a) for a in person["aliases"]) or "无"), ""])
            memories(person["memories"])
        lines.extend(["### 群体记忆", ""])
        memories(scope["group_memories"])
        if scope["blocked_topics"]:
            lines.extend(["### 已屏蔽自动写入的主题", ""])
            for topic in scope["blocked_topics"]:
                lines.append(f"- {md(topic['subject_id'])} / {md(topic['kind'])} / {md(topic['key'])}")
            lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, text: str):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".cdmem-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_views(directory: Path, data):
    atomic_write(directory / "memory.md", markdown(data))
    atomic_write(directory / "memory.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")
