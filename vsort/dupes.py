"""重複檔偵測（F17）：用檔案大小加部分內容雜湊（頭、中、尾各 1 MB）找出重複下載。

不存資料庫：每次執行時掃描工作區與收藏庫的檔案大小（很快），只有大小相同的才讀內容比對。
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

from vsort.config import Config

_CHUNK = 1024 * 1024


def partial_hash(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(_CHUNK))
        if size > 3 * _CHUNK:
            fh.seek(size // 2)
            h.update(fh.read(_CHUNK))
            fh.seek(size - _CHUNK)
            h.update(fh.read(_CHUNK))
    return h.hexdigest()


class DupIndex:
    """以大小分組的檔案索引，查詢時才計算雜湊並快取。"""

    def __init__(self, exts: set[str]):
        self.exts = exts
        self.by_size: dict[int, list[Path]] = defaultdict(list)
        self._hash: dict[str, str] = {}

    def add_tree(self, root: Path, exclude: Path | None = None) -> None:
        if not root.exists():
            return
        for dirpath, dirnames, names in os.walk(root):
            if exclude is not None and Path(dirpath).resolve() == exclude.resolve():
                dirnames[:] = []
                continue
            for n in names:
                p = Path(dirpath) / n
                if p.suffix.lower() in self.exts:
                    self.add(p)

    def add(self, p: Path) -> None:
        try:
            self.by_size[p.stat().st_size].append(p)
        except OSError:
            pass

    def _h(self, p: Path) -> str:
        key = os.path.normcase(str(p))
        if key not in self._hash:
            self._hash[key] = partial_hash(p)
        return self._hash[key]

    def find(self, p: Path) -> Path | None:
        """回傳和 p 內容相同的已知檔案（不含 p 自己）。"""
        try:
            size = p.stat().st_size
        except OSError:
            return None
        me = os.path.normcase(str(p.resolve()))
        same = [q for q in self.by_size.get(size, []) if os.path.normcase(str(q.resolve())) != me]
        if not same:
            return None
        h = self._h(p)
        for q in same:
            try:
                if self._h(q) == h:
                    return q
            except OSError:
                continue
        return None

    def groups(self) -> list[list[Path]]:
        out = []
        for size, paths in self.by_size.items():
            if len(paths) < 2:
                continue
            by_hash: dict[str, list[Path]] = defaultdict(list)
            for p in paths:
                try:
                    by_hash[self._h(p)].append(p)
                except OSError:
                    continue
            out.extend(g for g in by_hash.values() if len(g) > 1)
        return sorted(out, key=lambda g: str(g[0]))


def workspace_index(cfg: Config) -> DupIndex:
    idx = DupIndex(cfg.video_extensions)
    idx.add_tree(cfg.layout.workspace)
    if not _inside(cfg.layout.library, cfg.layout.workspace):
        idx.add_tree(cfg.layout.library, exclude=cfg.layout.workspace)
    return idx


def _inside(a: Path, b: Path) -> bool:
    try:
        a.resolve().relative_to(b.resolve())
        return True
    except ValueError:
        return False
