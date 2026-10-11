"""依畫風推薦作者（只當參考，不影響歸檔）。

做法：每部影片的 SigLIP 畫面向量就像「畫風指紋」，拿不知道作者的影片去和已知作者的影片比，
列出最像的前 3 位作者。參考資料包含：收藏庫裡有作者的影片，以及你標註過、放在作者資料夾裡的影片。

實測（你標過的 155 部、35 位作者、排除同一作品的差分）：最像的那位猜對 48%，前 3 名內 68%。
所以只適合當線索，影片本身仍歸到「作者未知」，等爬蟲或人工處理。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from vsort.config import Config


class StyleAuthorIndex:
    def __init__(self, cfg: Config):
        self.emb_dir = cfg.layout.workspace / "_cache" / "emb" / "siglip"
        self.keys: list[str] = []
        self.authors: list[str] = []
        self._build(cfg)

    def _build(self, cfg: Config) -> None:
        from vsort.analyze import file_key
        from vsort.naming import parse_final_name
        from vsort.process import _videos
        seen: set[str] = set()
        for v in _videos(cfg.layout.library, cfg):  # 收藏庫：檔名裡就有作者
            f = parse_final_name(v.name)
            if f and f.get("author"):
                try:
                    self._add(file_key(v), f["author"], seen)
                except OSError:
                    continue
        try:  # 你標註過的影片：作者取自資料夾
            from vsort.labeling import load_labels
            from vsort.process import make_parser
            parser = make_parser(cfg)
            for key, lab in load_labels(cfg).items():
                p = Path(lab["path"])
                parts = list(p.parent.parts[1:])  # 去掉磁碟代號；Videos、未分類這類通用名稱解析時會跳過
                a = parser.parse(p.name, parts).author
                if a:
                    self._add(key, a, seen)
        except Exception:
            pass
        if self.keys:
            X = np.stack([np.load(self.emb_dir / f"{k}.npy") for k in self.keys]).astype(np.float32)
            self.mean = X.mean(0)
            X -= self.mean
            self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def _add(self, key: str, author: str, seen: set[str]) -> None:
        if key in seen or not (self.emb_dir / f"{key}.npy").exists():
            return
        seen.add(key)
        self.keys.append(key)
        self.authors.append(author)

    def suggest(self, key: str, k: int = 3) -> list[tuple[str, float]]:
        p = self.emb_dir / f"{key}.npy"
        if not self.keys or not p.exists():
            return []
        q = np.load(p).astype(np.float32) - self.mean
        q /= np.linalg.norm(q) + 1e-9
        sims = self.X @ q
        best: dict[str, float] = {}
        for i in np.argsort(-sims):
            if self.keys[i] == key:
                continue
            a = self.authors[i]
            if a not in best:
                best[a] = float(sims[i])
            if len(best) >= k:
                break
        return [(a, round(s, 3)) for a, s in best.items()]


def export_unknown(cfg: Config) -> tuple[Path, int]:
    """列出收藏庫「作者未知」與人工確認區裡不知道作者的影片，附畫風相近的作者，給之後處理用。"""
    from vsort import sidecar as sc
    from vsort.analyze import UNKNOWN_AUTHOR_DIR
    from vsort.process import _videos
    rows = []
    for folder in (cfg.layout.library / UNKNOWN_AUTHOR_DIR, cfg.layout.review):
        for v in _videos(folder, cfg):
            d = sc.read(v) or {}
            if (d.get("info") or {}).get("author"):
                continue
            sugg = d.get("style_authors") or []
            rs = d.get("reverse_search") or {}
            clues = [f"{s}:{i}" for s, i in rs.get("sources", [])] + rs.get("titles", [])[:3] \
                if isinstance(rs, dict) else []
            rows.append([str(v), d.get("original_name", ""),
                         "、".join(f"{a}（{s:.2f}）" for a, s in sugg), "、".join(clues)])
    cfg.layout.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.layout.reports / "unknown_authors.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["影片", "原始檔名", "畫風相近的作者（相似度）", "以圖搜圖線索"])
        w.writerows(rows)
    return out, len(rows)
