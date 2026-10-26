"""Build an allowlisted AstrBot ZIP, excluding all memory, credentials and caches."""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["main.py", "__init__.py", "metadata.yaml", "_conf_schema.json", "requirements.txt", "README.md", "LICENSE"]


def build():
    output = ROOT / "dist" / "astrbot_plugin_chuanda_memory-v0.1.0.zip"
    output.parent.mkdir(exist_ok=True)
    paths = [ROOT / name for name in FILES]
    paths += sorted((ROOT / "chuanda_memory").glob("*.py"))
    paths += sorted((ROOT / "docs").glob("*.md"))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    print(output)
    print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


if __name__ == "__main__":
    build()
