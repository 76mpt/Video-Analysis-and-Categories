"""安全的檔案搬移與複製（F30）：同一顆硬碟直接改名，跨硬碟先複製、核對再刪原檔。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PART_SUFFIX = ".vsort-part"
_SAMPLE = 1024 * 1024


def part_path(dst: Path) -> Path:
    return dst.with_name(dst.name + PART_SUFFIX)


def _existing_ancestor(p: Path) -> Path:
    while not p.exists():
        p = p.parent
    return p


def same_volume(src: Path, dst: Path) -> bool:
    return os.stat(src).st_dev == os.stat(_existing_ancestor(dst)).st_dev


def _sample(path: Path, size: int) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(_SAMPLE)
        if size > 2 * _SAMPLE:
            fh.seek(size - _SAMPLE)
        tail = fh.read(_SAMPLE)
    return head + tail


def files_match(a: Path, b: Path) -> bool:
    """以大小加頭尾各 1 MB 內容核對兩個檔案是否相同。"""
    sa, sb = a.stat().st_size, b.stat().st_size
    return sa == sb and _sample(a, sa) == _sample(b, sb)


def copy_verified(src: Path, dst: Path) -> None:
    """複製到暫存名，核對無誤才改成正式檔名；dst 不可已存在。"""
    if dst.exists():
        raise FileExistsError(f"目的地已存在：{dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dst)
    if part.exists():
        part.unlink()
    shutil.copy2(src, part)
    if not files_match(src, part):
        part.unlink()
        raise OSError(f"複製後核對不符：{src}")
    os.rename(part, dst)


def move_file(src: Path, dst: Path) -> None:
    """搬移檔案；dst 不可已存在，絕不覆蓋。"""
    if dst.exists():
        raise FileExistsError(f"目的地已存在：{dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if same_volume(src, dst):
        os.rename(src, dst)  # Windows 上目的地存在時 rename 會失敗，不會覆蓋
    else:
        copy_verified(src, dst)
        os.remove(src)
