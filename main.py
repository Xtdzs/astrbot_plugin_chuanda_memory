"""AstrBot adapter. All durable memory lives outside this plugin's source tree."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart

from .chuanda_memory.client import RemoteBackend
from .chuanda_memory import __version__
from .chuanda_memory.engine import LocalBackend, MemoryEngine
from .chuanda_memory.extraction import CONTEXT_RULES, PERSONA
from .chuanda_memory.models import Observation, identity_namespace
from .chuanda_memory.store import MemoryStore

_extracting = ContextVar("chuanda_memory_extracting", default=False)
_EVENT_KEY = "chuanda_memory_observation_v1"


class ChuandaMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.backend = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    async def initialize(self):
        async with self._initialize_lock:
            if self.backend is not None or self._closed:
                return
            if self.config.get("mode", "local") == "remote":
                token = os.environ.get("CHUANDA_MEMORY_TOKEN", "") or self.config.get("service_token", "")
                self.backend = RemoteBackend(self.config.get("service_url", "http://127.0.0.1:8765"), token,
                                             timeout=float(self.config.get("service_timeout_seconds", 3)))
            else:
                configured = os.environ.get("CHUANDA_MEMORY_DATA_DIR", "") or self.config.get("data_dir", "")
                directory = Path(configured).expanduser().resolve() if configured else StarTools.get_data_dir("astrbot_plugin_chuanda_memory")
                source = Path(__file__).resolve().parent
                if directory == source or source in directory.parents:
                    raise ValueError("记忆目录不能位于插件代码目录内，请选择外置数据目录")
                store = await asyncio.to_thread(MemoryStore, directory, readable_background=True)
                await asyncio.to_thread(store.prune, int(self.config.get("raw_retention_days", 14)))
                engine = MemoryEngine(store, self._call_model, delay=float(self.config.get("batch_delay_seconds", 20)),
                                      batch_size=int(self.config.get("batch_size", 16)),
                                      timeout=float(self.config.get("extraction_timeout_seconds", 45)),
                                      retention_days=int(self.config.get("raw_retention_days", 14)))
                self.backend = LocalBackend(engine, recall_limit=int(self.config.get("recall_limit", 8)),
                                            char_budget=int(self.config.get("context_char_budget", 5000)),
                                            cooldown=float(self.config.get("joke_cooldown_hours", 6)) * 3600)
                engine.start()
            logger.info("串大记忆已加载（%s），不会在聊天中播报后台记忆更新。", self.config.get("mode", "local"))

    def _allowed(self, event):
        if not self.config.get("enabled", True) or _extracting.get():
            return False
        sender = str(event.get_sender_id() or "")
        if not sender or sender == str(event.get_self_id() or ""):
            return False
        group = str(event.get_group_id() or "")
        if not group:
            return bool(self.config.get("enable_private_memory", True))
        groups = [str(g) for g in self.config.get("allowed_groups", [])]
        return not groups or group in groups

    @staticmethod
    def _identity(event):
        namespace = identity_namespace(str(event.get_platform_name()), str(event.get_platform_id()), str(event.get_self_id() or ""))
        sender = str(event.get_sender_id())
        group = str(event.get_group_id() or "")
        return namespace, "group:" + group if group else "private:" + sender, sender

    async def _call_model(self, system, prompt, provider):
        provider = self.config.get("memory_provider_id", "") or provider
        if not provider:
            raise ValueError("No memory extraction provider is configured")
        guard = _extracting.set(True)
        try:
            result = await self.context.llm_generate(chat_provider_id=provider, prompt=prompt,
                                                     system_prompt=system, contexts=[], tools=None)
            return result.completion_text
        finally:
            _extracting.reset(guard)

    async def _observe(self, event):
        cached = event.get_extra(_EVENT_KEY)
        if cached is not None:
            return cached
        text = event.get_message_str().strip()
        command = text.lstrip("/").split(maxsplit=1)[0].casefold() if text else ""
        if not text or command in {"cdmem", "串大记忆", "串大忘记"}:
            return None
        # Keep only top-level plain text. Quotes/forwarded messages are not self-report evidence.
        plain = [part.text for part in event.get_messages() if getattr(part, "type", "") == "Plain"]
        if plain:
            text = " ".join(plain).strip()
        if not text:
            return None
        ns, scope, sender = self._identity(event)
        if sender == "@group":
            return None
        provider = self.config.get("memory_provider_id", "")
        if not provider and isinstance(self.backend, LocalBackend):
            try:
                provider = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            except Exception:
                provider = ""
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        if not message_id:
            # Event-local UUID: repeated equal messages are different observations.
            message_id = "event-" + uuid.uuid4().hex
        observation = Observation(ns, scope, message_id, sender, str(event.get_sender_name() or "")[:128],
                                  text[:6000], str(provider or ""), time.time())
        await self.backend.observe(observation)
        event.set_extra(_EVENT_KEY, observation)
        return observation

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def observe_message(self, event: AstrMessageEvent):
        if not self._allowed(event) or not self.config.get("capture_all_messages", True):
            return
        try:
            await self.initialize()
            await self._observe(event)
        except Exception as exc:
            logger.warning("串大记忆观察暂不可用：%s", type(exc).__name__)

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._allowed(event) or getattr(req, "_chuanda_memory_injected", False):
            return
        try:
            await self.initialize()
            await self._observe(event)
            ns, scope, sender = self._identity(event)
            context = await self.backend.context(ns, scope, sender, event.get_message_str()[:6000])
            if self.config.get("enable_chuanda_persona", True):
                # Stable persona only. Dynamic memories stay in temporary user content.
                if PERSONA not in (req.system_prompt or ""):
                    req.system_prompt = (req.system_prompt or "") + "\n" + PERSONA
            text = CONTEXT_RULES + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())
            req._chuanda_memory_injected = True
        except Exception as exc:
            # A broken memory service must never block ordinary QQ replies.
            logger.warning("串大记忆召回暂不可用，继续正常回复：%s", type(exc).__name__)

    @filter.command("cdmem")
    async def memory_command(self, event: AstrMessageEvent, action: str = "help", target: str = ""):
        event.stop_event()
        if not self._allowed(event):
            group = str(event.get_group_id() or "")
            if not self.config.get("enabled", True):
                reason = "plugin_disabled"
            elif group:
                reason = "group_not_allowed"
            else:
                reason = "private_memory_disabled"
            if action.casefold() in {"status", "help"}:
                text = (f"CDMem: {__version__}\nPlatform: {event.get_platform_name()}\n"
                        f"Group ID: {group or '(private)'}\nID: {event.get_sender_id()}\nCapture: blocked ({reason})\n"
                        "本指令只显示当前事件身份，未读取或采集记忆。\n"
                        "群未获准时，请把上面的 Group ID 填入 allowed_groups；QQ 官方接口使用 OpenID，不是 QQ 群号。")
            else:
                text = f"CDMem blocked ({reason}). Use /cdmem status."
            yield event.plain_result(text)
            return
        try:
            await self.initialize()
            ns, scope, sender = self._identity(event)
            action = action.casefold()
            if action == "help":
                text = ("/cdmem status\n/cdmem show [user_id]\n/cdmem group\n"
                        "/cdmem pause\n/cdmem resume\n/cdmem clear\n/cdmem forget <memory_id>\n"
                        "show 默认查看自己，其他用户档案仅管理员可看。clear 清空当前范围内与你相关的记忆并暂停采集。")
            elif action == "pause":
                await self.backend.pause(ns, scope, sender)
                text = "已暂停当前范围内对你的记忆采集。用 /cdmem resume 恢复。"
            elif action == "resume":
                await self.backend.resume(ns, scope, sender)
                text = "已恢复当前范围内对你的记忆采集。"
            elif action == "clear":
                await self.backend.pause(ns, scope, sender, erase=True)
                text = "已清空当前范围内与你相关的记忆和来源消息，并暂停采集。用 /cdmem resume 恢复。"
            elif action == "status":
                profile = await self.backend.profile(ns, scope, sender)
                text = (f"CDMem: {__version__}\nPlatform: {event.get_platform_name()}\nGroup ID: {event.get_group_id() or '(private)'}\n"
                        f"ID: {sender}\nScope: {scope}\nCapture: {'paused' if profile['paused'] else 'active'}\nMemories: {len(profile['memories'])}")
            elif action in {"show", "group"}:
                subject = "@group" if action == "group" else (target or sender)
                if subject != sender and action != "group" and not event.is_admin():
                    text = "这里只能查看你自己的档案。"
                else:
                    profile = await self.backend.profile(ns, scope, subject)
                    rows = [f"{m['id']} [{m['kind']}] {m['content'][:180]}" for m in profile["memories"][:15]]
                    text = "\n".join(rows) if rows else "当前聊天范围内还没有这个人的有效人物记忆。"
            elif action == "forget":
                if not target:
                    text = "Usage: /cdmem forget <memory_id>"
                else:
                    text = await self._forget_text(event, target)
            else:
                text = "Unknown command. Use /cdmem help."
        except Exception as exc:
            logger.warning("串大记忆管理失败：%s", type(exc).__name__)
            text = "记忆服务暂时不可用，请稍后再试。"
        yield event.plain_result(text)

    async def _forget_text(self, event: AstrMessageEvent, memory_id: str):
        try:
            ns, scope, sender = self._identity(event)
            deleted = await self.backend.forget(ns, scope, sender, memory_id, admin=event.is_admin())
            return "已删除这条记忆，并阻止同主题自动重新写入。" if deleted else "当前聊天范围内没有这条记忆。"
        except PermissionError:
            return "只能删除与你相关的记忆；其他记忆由 AstrBot 管理员管理。"

    async def terminate(self):
        self._closed = True
        if self.backend is not None:
            await self.backend.close()
