"""每部影片旁的資訊檔 <影片檔名>.vsort.json（純文字，跟著影片搬移）。

不用資料庫：檔案在哪個資料夾代表它的狀態，影片旁的資訊檔記錄解析、查詢與 AI 判定的結果，
後面的步驟直接讀它。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SUFFIX = ".vsort.json"


def path_for(video: Path) -> Path:
    return video.with_name(video.name + SUFFIX)


def preview_for(video: Path) -> Path:
    """送人工確認時附的關鍵幀預覽圖。"""
    return video.with_name(video.name + ".vsort.jpg")


def is_sidecar(path: Path) -> bool:
    return path.name.endswith(SUFFIX)


def read(video: Path) -> dict | None:
    p = path_for(video)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_at(path: Path, data: dict) -> None:
    """原子寫入：先寫暫存檔再取代，斷電也不會留下寫一半的檔案。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write(video: Path, data: dict) -> None:
    write_at(path_for(video), data)
