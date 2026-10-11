"""挑選與掃描影片（F02、F29）：只處理使用者指定的檔案或資料夾。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from vsort.config import Config


@dataclass
class Selected:
    path: Path
    root: Path | None  # 若來自挑選的資料夾，記下該資料夾（用來保留結構）


@dataclass
class ScanResult:
    files: list[Selected] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    non_video_count: int = 0


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_file(path: Path, cfg: Config, now: float | None = None) -> str | None:
    """回傳略過原因；可處理則回傳 None。"""
    ext = path.suffix.lower()
    if ext in cfg.temp_extensions:
        return "下載中的暫存檔"
    for temp_ext in cfg.temp_extensions:
        if Path(str(path) + temp_ext).exists():
            return "旁邊有下載暫存檔，可能還在下載"
    if ext not in cfg.video_extensions:
        return "不是影片副檔名"
    st = path.stat()
    if st.st_size == 0:
        return "檔案大小為 0"
    age = (now or time.time()) - st.st_mtime
    if age < cfg.data["min_file_age_seconds"]:
        return f"{int(age)} 秒前才修改，可能仍在寫入"
    return None


def scan(inputs: list[Path], cfg: Config, recursive: bool = True,
         exclude: list[Path] | None = None) -> ScanResult:
    """展開使用者挑選的路徑。exclude 內的資料夾（例如工作區本身）會被略過。"""
    result = ScanResult()
    seen: set[str] = set()
    exclude = exclude or []
    now = time.time()

    def add(p: Path, root: Path | None, explicit: bool) -> None:
        key = os.path.normcase(str(p.resolve()))
        if key in seen:
            return
        seen.add(key)
        if any(_inside(p, ex) for ex in exclude):
            result.skipped.append((p, "已經在工作區裡"))
            return
        reason = check_file(p, cfg, now)
        if reason == "不是影片副檔名" and not explicit:
            result.non_video_count += 1
        elif reason:
            result.skipped.append((p, reason))
        else:
            result.files.append(Selected(p, root))

    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            result.skipped.append((p, "找不到這個路徑"))
        elif p.is_file():
            add(p, None, explicit=True)
        elif p.is_dir():
            if recursive:
                for dirpath, dirnames, filenames in os.walk(p):
                    dirnames.sort()
                    for name in sorted(filenames):
                        add(Path(dirpath) / name, p, explicit=False)
            else:
                for child in sorted(p.iterdir()):
                    if child.is_file():
                        add(child, p, explicit=False)
    return result
