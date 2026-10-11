import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from vsort.config import PROJECT_ROOT, load_config


@pytest.fixture
def cfg(tmp_path):
    # 測試不碰真正的字典，也不帶入你的標註與待確認清單
    shutil.copytree(PROJECT_ROOT / "dict", tmp_path / "dict",
                    ignore=shutil.ignore_patterns("style_labels.csv", "authors_pending.csv", "*.tmp"))
    conf = {"workspace_dir": str(tmp_path / "ws"), "min_file_age_seconds": 60,
            "dictionary_dir": str(tmp_path / "dict"), "lookup": {"enabled": False}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(conf), encoding="utf-8")
    c = load_config(path)
    c.layout.ensure()
    return c


def make_video(path: Path, size: int = 2048, age: float = 3600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    old = time.time() - age
    os.utime(path, (old, old))
    return path
