"""Download public host sources for API verification; never reads local credentials."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / ".reference"
FILES = [
    "astrbot/core/provider/entities.py",
    "astrbot/core/agent/message.py",
    "astrbot/core/platform/astr_message_event.py",
    "astrbot/core/star/context.py",
    "astrbot/core/star/star_tools.py",
    "astrbot/api/star/__init__.py",
    "astrbot/api/__init__.py",
    "astrbot/core/star/register/__init__.py",
    "astrbot/core/star/register/star_handler.py",
    "astrbot/api/event/filter/__init__.py",
    "astrbot/core/star/filter/__init__.py",
    "astrbot/core/star/filter/event_message_type.py",
    "astrbot/core/message/components.py",
    "docs/zh/dev/star/guides/listen-message-event.md",
]

def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "chuanda-memory-dev"}), timeout=30) as response:
        return response.read()

if __name__ == "__main__":
    commit = (ROOT / "revision.txt").read_text().strip() if (ROOT / "revision.txt").exists() else json.loads(fetch("https://api.github.com/repos/AstrBotDevs/AstrBot/commits/master"))["sha"]
    ROOT.mkdir(exist_ok=True)
    (ROOT / "revision.txt").write_text(commit, encoding="utf-8")
    for name in FILES:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            target.write_bytes(fetch(f"https://raw.githubusercontent.com/AstrBotDevs/AstrBot/{commit}/{name}"))
            print(name)
        except Exception as exc:
            print(f"Unavailable: {name}: {type(exc).__name__}")
    print("AstrBot revision:", commit)
