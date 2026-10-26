"""Static contract check against public AstrBot sources fetched by fetch_reference.py."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".reference"


def tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def method(path, class_name, method_name):
    cls = next(n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name)


def check():
    request = next(n for n in tree("astrbot/core/provider/entities.py").body if isinstance(n, ast.ClassDef) and n.name == "ProviderRequest")
    fields = {n.target.id for n in request.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert {"system_prompt", "extra_user_content_parts"} <= fields
    method("astrbot/core/agent/message.py", "ContentPart", "mark_as_temp")
    for name in ("get_platform_id", "get_platform_name", "get_sender_id", "get_sender_name", "get_self_id",
                 "get_group_id", "get_message_str", "get_messages", "get_extra", "set_extra", "is_admin", "stop_event", "plain_result"):
        method("astrbot/core/platform/astr_message_event.py", "AstrMessageEvent", name)
    generate = method("astrbot/core/star/context.py", "Context", "llm_generate")
    kwargs = {n.arg for n in generate.args.kwonlyargs}
    assert {"chat_provider_id", "prompt", "system_prompt", "contexts", "tools"} <= kwargs
    method("astrbot/core/star/context.py", "Context", "get_current_chat_provider_id")
    method("astrbot/core/star/star_tools.py", "StarTools", "get_data_dir")
    star_exports = (ROOT / "astrbot/api/star/__init__.py").read_text(encoding="utf-8")
    assert "StarTools" in star_exports and "Context" in star_exports
    filters = (ROOT / "astrbot/api/event/filter/__init__.py").read_text(encoding="utf-8")
    assert "as on_llm_request" in filters and "as event_message_type" in filters and "as command" in filters
    handler_tree = tree("astrbot/core/star/register/star_handler.py")
    for name in ("register_event_message_type", "register_command"):
        func = next(n for n in handler_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert func.args.kwarg or any(a.arg == "priority" for a in func.args.kwonlyargs + func.args.args)
    print("AstrBot public API contract verified at", (ROOT / "revision.txt").read_text().strip())


if __name__ == "__main__":
    check()
