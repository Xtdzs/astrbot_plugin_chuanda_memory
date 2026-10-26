import importlib.util
import json
import zipfile
import unittest
from pathlib import Path


class DistributionTests(unittest.TestCase):
    def test_installable_archive_has_no_runtime_data_or_credentials(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("release_builder", root / "tools/build_release.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        path = builder.build()
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            self.assertTrue({"main.py", "metadata.yaml", "_conf_schema.json", "chuanda_memory/store.py", "chuanda_memory/server.py"} <= names)
            self.assertFalse(any(name.endswith((".db", ".sqlite3", ".env")) or name.startswith((".reference/", ".scratch/", ".venv/")) for name in names))
            for name in names:
                if name.endswith(".py"):
                    compile(archive.read(name), name, "exec")
            schema = json.loads(archive.read("_conf_schema.json"))
            self.assertEqual(schema["mode"]["default"], "local")
            self.assertIn("v0.1.0", archive.read("metadata.yaml").decode())
