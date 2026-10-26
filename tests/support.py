import tempfile
from pathlib import Path


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / ".scratch" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(dir=root)
    assert Path(temporary.name).resolve().is_relative_to(root.resolve())
    return temporary
