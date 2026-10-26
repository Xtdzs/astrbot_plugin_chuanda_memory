from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field


def identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class Observation:
    namespace: str
    scope: str
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    provider_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        for name in ("namespace", "scope", "message_id", "sender_id"):
            identifier(getattr(self, name), name)
        if self.sender_id == "@group":
            raise ValueError("Reserved sender ID")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 6000:
            raise ValueError("Message text must contain 1–6000 characters")
        if not isinstance(self.sender_name, str) or len(self.sender_name) > 128:
            raise ValueError("Invalid sender name")
        if not isinstance(self.provider_id, str) or len(self.provider_id) > 256:
            raise ValueError("Invalid provider ID")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError("Invalid timestamp")

    def to_dict(self):
        return asdict(self)


def identity_namespace(platform: str, instance: str, bot_id: str) -> str:
    """Never infer identity from a nickname or from text supplied by a model."""
    return json.dumps([platform, instance, bot_id], ensure_ascii=False, separators=(",", ":"))
