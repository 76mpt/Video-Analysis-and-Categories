"""oreno3D：iwara 上 MMD／3D 影片的索引站，不放影片，但保留標題、作者、
原作、角色、標籤與 iwara 連結；iwara 上已刪除的影片資料也常留著。

沒有 API，用搜尋頁（/search?keyword=）與影片頁（/movies/<編號>）的 HTML。
用法：先用檔名標題搜 → 搜尋結果已有標題與作者，可先比對 → 對中的再開影片頁拿 iwara ID 與標籤。
"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from vsort.net import Http, NetError

BASE = "https://oreno3d.com"
CACHE_TTL = 7 * 86400

_CARD = re.compile(r'<a href="https://oreno3d\.com/movies/(?P<id>\d+)"[^>]*>.*?<h2 class="box-h2">(?P<title>.*?)</h2>'
                   r'.*?<div class="box-text-in">(?P<author>.*?)</div>', re.S)
_IWARA = re.compile(r'https://(?:www\.)?iwara\.tv/videos?/([A-Za-z0-9]{8,20})')


def _text(s: str) -> str:
    s = re.sub(r'<i class="material-icons[^"]*">.*?</i>', "", s, flags=re.S)  # 圖示文字（face 等）
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def _section(page: str, label: str, kind: str) -> list[str]:
    """影片頁「作成者：」「原作：」「キャラクター：」區塊裡的連結文字。"""
    i = page.find(f"{label}：</h2>")
    if i < 0:
        return []
    end = page.find("</section>", i)
    return [_text(m) for m in re.findall(rf'href="https://oreno3d\.com/{kind}/\d+"[^>]*>(.*?)</a>', page[i:end], re.S)]


@dataclass
class OrenoMovie:
    movie_id: str
    title: str
    author: str | None = None
    iwara_id: str | None = None
    origins: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    published: str | None = None


class Oreno3D:
    def __init__(self, cache_dir: Path, http: Http | None = None, online: bool = True):
        self.cache_dir = Path(cache_dir) / "oreno3d"
        self.http = http or Http(min_interval=2.0, timeout=30)
        self.online = online

    def _cached(self, key: str, fetch):
        f = self.cache_dir / (re.sub(r"[^\w.-]", "_", key)[:150] + ".json")
        if f.exists() and time.time() - f.stat().st_mtime < CACHE_TTL:
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if not self.online:
            return None
        data = fetch()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def search(self, keyword: str, page: int = 1) -> list[dict]:
        """回傳 [{id, title, author}]，依網站的相關度排序。"""
        def fetch():
            t = self.http.text(f"{BASE}/search?keyword={quote(keyword)}&page={page}")
            return [{"id": m["id"], "title": _text(m["title"]), "author": _text(m["author"])}
                    for m in _CARD.finditer(t)]
        return self._cached(f"search_{keyword}_{page}", fetch) or []

    def movie(self, movie_id: str) -> OrenoMovie | None:
        def fetch():
            p = self.http.text(f"{BASE}/movies/{movie_id}")
            title = re.search(r'class="video-h1">(.*?)</h1>', p, re.S)
            iw = _IWARA.search(p)
            date = re.search(r"calendar_month</i>\s*(?:<[^>]+>\s*)*([\d-]{10})", p) or \
                re.search(r"(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}", p)
            authors = _section(p, "作成者", "authors")
            return {"movie_id": movie_id, "title": _text(title.group(1)) if title else "",
                    "author": authors[0] if authors else None, "iwara_id": iw.group(1) if iw else None,
                    "origins": _section(p, "原作", "origins"), "characters": _section(p, "キャラクター", "characters"),
                    "tags": [_text(t) for t in re.findall(r'href="https://oreno3d\.com/tags/\d+"[^>]*>(.*?)</a>', p, re.S)],
                    "published": date.group(1) if date else None}
        try:
            d = self._cached(f"movie_{movie_id}", fetch)
        except NetError as e:
            if e.kind == "notfound":
                return None
            raise
        return OrenoMovie(**d) if d else None

    def find(self, titles: list[str], author: str | None = None, min_score: float = 0.9) -> tuple[OrenoMovie, float] | None:
        """用檔名標題搜，標題相似（作者同名可加分）就回傳影片資料與分數。"""
        from vsort.hints import _core, similarity

        scored: list[tuple[float, bool, dict]] = []
        seen = set()
        for t in titles:
            core = _core(t)
            words = sorted(core.split(), key=len, reverse=True)
            # 網站搜尋是關鍵字比對，長標題常搜不到：先整句，再用最長的兩個詞
            for q in dict.fromkeys([core, " ".join(words[:2]), words[0] if words else ""]):
                if len(q) < 3:
                    continue
                for r in self.search(q)[:20]:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    same_author = bool(author and r.get("author") and similarity(author, r["author"]) > 0.8)
                    scored.append((similarity(t, r["title"]) + (0.1 if same_author else 0), same_author, r))
                if any(x[0] >= 1.0 for x in scored):
                    break
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        s, same_author, r = scored[0]
        if s < min_score:
            return None
        # 同名作品很多（翻唱、同一首歌）：作者對不上又有幾乎一樣的候選 → 不採用
        rivals = [x for x in scored[1:] if x[0] >= s - 0.05]
        if rivals and not same_author:
            return None
        mv = self.movie(r["id"])
        return (mv, round(min(s, 1.0), 3)) if mv else None
