"""檔名線索搜尋：檔名沒有網站 ID 時，用作者、標題、日期去來源網站搜尋，猜出原始貼文。

做法（由準到不準）：
1. 作者資料夾 → 找 kemono／pawchive 鏡像站上同名的創作者 → 掃他的貼文：
   附件檔名和影片檔名一樣就是確定命中；否則比標題相似度與日期。
2. 作者資料夾 → 找 iwara 上同名的帳號 → 掃他的影片標題。
3. 不知道作者時：用標題在 iwara 與 kemono 全站搜尋，要標題幾乎一樣才採用。

找到的 (網站, 貼文ID) 再交給 Lookup.fetch 查完整資訊（作者、標籤）。
所有搜尋結果快取在 Workspace\\_cache\\hints。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

from vsort.dictionary import normalize
from vsort.net import Http, NetError
from vsort.parser import NOISE, Parsed

CACHE_TTL = 7 * 86400
MAX_PAGES = 12            # 每位創作者最多掃幾頁（kemono 一頁 50 篇、iwara 一頁 50 部）
ACCEPT = 0.75             # 分數達到這個才算命中
MIRRORS = ("kemono.cr", "pawchive.st")
MIRROR_SERVICES = ("fanbox", "fantia", "patreon")

_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{2})[.](\d{1,2})[.](\d{1,2})(?!\d)"), "yy.m.d"),
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)"), "yymmdd"),
]
_STOP = re.compile(r"\b(?:iwara|mmd|ver(?:sion)?|loop|bonus|mp4|vr ?180|lr|sbs|"
                   r"edit|normal|dance)\b", re.I)


@dataclass
class Hints:
    author: str | None
    titles: list[str]                  # 檔名標題＋下層資料夾名稱（常是作品標題）
    stem: str                          # 原檔名（不含副檔名）
    rest: str                          # 去掉隨機碼前綴的檔名
    date: date | None = None


@dataclass
class HintMatch:
    site: str                          # fanbox / fantia / patreon / iwara
    source_id: str
    score: float
    reason: str                        # 例如「附件檔名相同」
    title: str = ""
    author_name: str | None = None
    account_id: str | None = None
    via: str = ""                      # kemono.cr / pawchive.st / iwara
    others: list[tuple[str, str, float]] = field(default_factory=list)  # 次佳候選


def parse_date(text: str) -> date | None:
    for rx, kind in _DATE_PATTERNS:
        for m in rx.finditer(text):
            a, b, c = (int(x) for x in m.groups())
            y = a if kind == "ymd" else 2000 + a
            try:
                d = date(y, b, c)
            except ValueError:
                continue
            if date(2010, 1, 1) <= d <= date.today():
                return d
    return None


def extract_hints(path: str | Path, parsed: Parsed) -> Hints:
    p = Path(path)
    stem = p.stem
    rest = re.sub(r"^(?:[0-9a-f]{8}|\d{7,9})_", "", stem)
    titles = [t for t in [parsed.title, *parsed.folder_hints] if t and len(_core(t)) >= 3]
    d = None
    if parsed.date:
        d = datetime.strptime(parsed.date, "%Y-%m-%d").date()
    for text in [stem, *parsed.folder_hints]:
        d = d or parse_date(text)
    return Hints(author=parsed.author, titles=titles, stem=stem, rest=rest, date=d)


def _core(s: str) -> str:
    """比對用：去掉解析度、iwara、mmd 這類每部都有的字。"""
    t = normalize(s)
    t = NOISE.sub(" ", t)
    t = _STOP.sub(" ", t)
    t = re.sub(r"[\[\]【】()（）「」『』~〜!！?？,、.。:：#＃&＆+＋|/\\\-]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    a, b = _core(a), _core(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and short in long_:
        return 0.75 + 0.2 * len(short) / len(long_)
    r = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    if len(ta) >= 2 and len(tb) >= 2:
        r = max(r, len(ta & tb) / min(len(ta), len(tb)) * 0.9)
    return r


def _same_file(name: str, h: Hints) -> bool:
    n = Path(name).stem
    nr = re.sub(r"^(?:[0-9a-f]{8}|\d{7,9})_", "", n)
    a = {normalize(h.stem), normalize(h.rest)}
    return bool({normalize(n), normalize(nr)} & a) and len(normalize(h.rest)) >= 3


def _date_bonus(published: str | None, d: date | None) -> float:
    if not published or not d:
        return 0.0
    try:
        pd = datetime.fromisoformat(published[:10]).date()
    except ValueError:
        return 0.0
    gap = abs((pd - d).days)
    return 0.1 if gap <= 3 else 0.05 if gap <= 14 else -0.05 if gap > 120 else 0.0


class HintSearcher:
    def __init__(self, cache_dir: Path, http: Http | None = None, mirrors=MIRRORS, online: bool = True):
        self.cache_dir = Path(cache_dir) / "hints"
        self.http = http or Http(min_interval=1.0, timeout=60)
        self.mirrors = list(mirrors)
        self.online = online
        self._creators: dict[str, dict[str, list[dict]]] = {}

    # ---------- 快取 ----------

    def _cached(self, key: str, fetch, ttl: float = CACHE_TTL):
        f = self.cache_dir / (re.sub(r"[^\w.-]", "_", key)[:150] + ".json")
        if f.exists() and time.time() - f.stat().st_mtime < ttl:
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

    def _mjson(self, mirror: str, path: str):
        return self.http.json(f"https://{mirror}/api/v1/{path}", headers={"Accept": "text/css"})

    # ---------- 鏡像站（kemono／pawchive）----------

    def mirror_creators(self, mirror: str, name: str) -> list[dict]:
        if mirror not in self._creators:
            raw = self._cached(f"{mirror}_creators", lambda: [
                {"id": c["id"], "name": c["name"], "service": c["service"], "public_id": c.get("public_id")}
                for c in self._mjson(mirror, "creators")
                if c.get("service") in MIRROR_SERVICES]) or []
            idx: dict[str, list[dict]] = {}
            for c in raw:
                for k in {normalize(c["name"]), normalize(c.get("public_id") or "")} - {""}:
                    idx.setdefault(k, []).append(c)
            self._creators[mirror] = idx
        return self._creators[mirror].get(normalize(name), [])

    def mirror_posts(self, mirror: str, service: str, uid: str) -> list[dict]:
        def fetch():
            out = []
            for page in range(MAX_PAGES):
                try:
                    batch = self._mjson(mirror, f"{service}/user/{uid}/posts?o={page * 50}")
                except NetError as e:
                    if e.kind == "notfound" or out:
                        break
                    raise
                if not isinstance(batch, list) or not batch:
                    break
                out += [{"id": p["id"], "title": p.get("title") or "", "published": p.get("published"),
                         "files": [x["name"] for x in [p.get("file") or {}, *(p.get("attachments") or [])]
                                   if x.get("name")]} for p in batch]
                if len(batch) < 50:
                    break
            return out
        return self._cached(f"{mirror}_{service}_{uid}", fetch) or []

    # ---------- 官方：fantia（公開 API 有每個檔案的原始檔名）----------

    def _fantia_token(self) -> dict:
        if not getattr(self, "_ftok", None):
            html = self.http.text("https://fantia.jp/")
            m = re.search(r'name="csrf-token" content="([^"]+)"', html)
            if not m:
                raise NetError("other", "fantia 頁面格式變了")
            self._ftok = {"X-CSRF-Token": m.group(1), "X-Requested-With": "XMLHttpRequest"}
        return self._ftok

    def fantia_post(self, pid: str) -> dict | None:
        def fetch():
            try:
                post = self.http.json(f"https://fantia.jp/api/v1/posts/{pid}",
                                      headers={**self._fantia_token(), "Referer": f"https://fantia.jp/posts/{pid}"}
                                      ).get("post") or {}
            except NetError as e:
                if e.kind in ("notfound", "other"):
                    return {"id": pid, "title": "", "files": [], "missing": True}
                raise
            fc = post.get("fanclub") or {}
            files = [c.get("filename") for c in post.get("post_contents") or [] if c.get("filename")]
            files += [c.get("title") for c in post.get("post_contents") or [] if c.get("title")]
            published = None
            try:
                published = datetime.strptime(post.get("posted_at") or "", "%a, %d %b %Y %H:%M:%S %z").isoformat()
            except ValueError:
                pass
            return {"id": pid, "title": post.get("title") or "", "published": published, "files": files,
                    "author": fc.get("creator_name") or fc.get("name"), "uid": str(fc.get("id") or "")}
        return self._cached(f"fantia_post_{pid}", fetch, ttl=365 * 86400)

    def fantia_post_ids(self, fid: str, page: int) -> list[str]:
        def fetch():
            html = self.http.text(f"https://fantia.jp/fanclubs/{fid}/posts?page={page}")
            return list(dict.fromkeys(re.findall(r'class="link-block" href="/posts/(\d+)"', html)))
        return self._cached(f"fantia_list_{fid}_{page}", fetch, ttl=86400) or []

    def fantia_scan(self, fid: str, h: Hints, max_pages: int = 15):
        """由新到舊掃 fanclub 的貼文；遇到檔名完全一樣就提早停。"""
        for page in range(1, max_pages + 1):
            ids = self._retry(self.fantia_post_ids, fid, page)
            for pid in ids:
                post = self._retry(self.fantia_post, pid)
                if post and not post.get("missing"):
                    yield post
            if len(ids) < 20:
                break

    # ---------- 官方：fanbox（只有標題）----------

    _FB = {"Origin": "https://www.fanbox.cc", "Referer": "https://www.fanbox.cc/"}

    def fanbox_creators(self, name: str) -> list[str]:
        d = self._cached(f"fanbox_search_{normalize(name)}", lambda: self.http.json(
            f"https://api.fanbox.cc/creator.search?q={quote(name)}&page=0", headers=self._FB)) or {}
        b = d.get("body") or []
        b = b.get("creators", []) if isinstance(b, dict) else b
        n = normalize(name)
        return [c["creatorId"] for c in b
                if n in (normalize(c.get("creatorId") or ""), normalize((c.get("user") or {}).get("name") or ""))][:2]

    def fanbox_posts(self, handle: str, max_pages: int = 30) -> list[dict]:
        def fetch():
            d = self.http.json(f"https://api.fanbox.cc/post.paginateCreator?creatorId={handle}", headers=self._FB)
            b = d.get("body") or []
            urls = b.get("pageUrls", []) if isinstance(b, dict) else b
            out = []
            for u in urls[:max_pages]:
                b2 = self.http.json(u, headers=self._FB).get("body") or []
                items = b2.get("posts") or b2.get("items") or [] if isinstance(b2, dict) else b2
                out += [{"id": p["id"], "title": p.get("title") or "", "published": p.get("publishedDatetime"),
                         "author": (p.get("user") or {}).get("name"), "uid": (p.get("user") or {}).get("userId"),
                         "files": []} for p in items]
            return out
        return self._cached(f"fanbox_posts_{handle}", fetch) or []

    # ---------- iwara ----------

    def iwara_users(self, name: str) -> list[dict]:
        d = self._cached(f"iwara_users_{normalize(name)}", lambda: self.http.json(
            f"https://api.iwara.tv/search?type=users&query={quote(name)}&page=0")) or {}
        n = normalize(name)
        return [u for u in d.get("results") or []
                if n in (normalize(u.get("name") or ""), normalize(u.get("username") or ""))][:3]

    def iwara_user_videos(self, uid: str) -> list[dict]:
        def fetch():
            out = []
            for page in range(MAX_PAGES):
                d = self.http.json(f"https://api.iwara.tv/videos?rating=all&user={uid}&limit=50&page={page}&sort=date")
                res = d.get("results") or []
                out += [{"id": v["id"], "title": v.get("title") or "", "published": v.get("createdAt"),
                         "author": (v.get("user") or {}).get("name"), "files": []} for v in res]
                if len(res) < 50:
                    break
            return out
        return self._cached(f"iwara_videos_{uid}", fetch) or []

    def iwara_search(self, title: str) -> list[dict]:
        d = self._cached(f"iwara_search_{normalize(title)}", lambda: self.http.json(
            f"https://api.iwara.tv/search?type=videos&query={quote(title)}&page=0")) or {}
        return [{"id": v["id"], "title": v.get("title") or "", "published": v.get("createdAt"),
                 "author": (v.get("user") or {}).get("name"),
                 "username": (v.get("user") or {}).get("username"), "files": []}
                for v in d.get("results") or []]

    def mirror_search(self, mirror: str, title: str) -> list[dict]:
        d = self._cached(f"{mirror}_search_{normalize(title)}",
                         lambda: self._mjson(mirror, f"posts?q={quote(title)}")) or {}
        return [{"id": p["id"], "title": p.get("title") or "", "published": p.get("published"),
                 "service": p.get("service"), "user": p.get("user"), "files": []}
                for p in (d.get("posts") if isinstance(d, dict) else d) or []
                if p.get("service") in MIRROR_SERVICES]

    @staticmethod
    def _retry(fn, *args, waits=(15, 60)):
        """fantia 連續查太多會暫時擋（429），等一下再試。"""
        for w in (*waits, None):
            try:
                return fn(*args)
            except NetError as e:
                if e.kind != "retry" or w is None:
                    raise
                time.sleep(w)

    @staticmethod
    def _safe(fn, *args) -> list:
        try:
            return fn(*args) or []
        except NetError:
            return []

    # ---------- 評分 ----------

    @staticmethod
    def score(post: dict, h: Hints) -> tuple[float, str]:
        if any(_same_file(f, h) for f in post.get("files") or []):
            return 1.0, "附件檔名相同"
        best = max((similarity(t, post["title"]) for t in h.titles), default=0.0)
        if best < 0.5:
            return best, ""
        return min(best + _date_bonus(post.get("published"), h.date), 0.99), \
            f"標題相似 {best:.2f}" + ("＋日期接近" if _date_bonus(post.get("published"), h.date) > 0 else "")

    def _pick(self, scored: list[tuple[float, str, str, str, dict]], need: float) -> HintMatch | None:
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        s, why, site, via, post = scored[0]
        if s < need:
            return None
        # 兩個不同貼文分數差不多 → 不確定，不採用（附件檔名相同例外）
        rivals = [x for x in scored[1:] if x[4]["id"] != post["id"] and x[0] >= s - 0.03]
        if rivals and s < 1.0:
            return None
        return HintMatch(site=site, source_id=str(post["id"]), score=round(s, 3), reason=why,
                         title=post["title"], author_name=post.get("author"), via=via,
                         account_id=post.get("user") or post.get("uid"),
                         others=[(x[2], str(x[4]["id"]), round(x[0], 3)) for x in scored[1:4]])

    # ---------- 對外 ----------

    def search(self, h: Hints, use_global: bool = True) -> HintMatch | None:
        scored = []

        def add(post, site, via):
            s, why = self.score(post, h)
            scored.append((s, why, site, via, post))
            return s >= 1.0

        if h.author:
            # 1. fantia 官方：fanclub ID 從鏡像站的創作者清單查（ID 與官方相同）
            fids = {c["id"] for m in self.mirrors for c in self._safe(self.mirror_creators, m, h.author)
                    if c["service"] == "fantia"}
            for fid in fids:
                try:
                    for post in self.fantia_scan(fid, h):
                        if add(post, "fantia", "fantia"):
                            return self._pick(scored, ACCEPT)
                except NetError:
                    pass
            # 2. 鏡像站（kemono、pawchive）：附件檔名
            for mirror in self.mirrors:
                for c in self._safe(self.mirror_creators, mirror, h.author):
                    for post in self._safe(self.mirror_posts, mirror, c["service"], c["id"]):
                        if add({**post, "author": c["name"], "uid": c["id"]}, c["service"], mirror):
                            return self._pick(scored, ACCEPT)
            # 3. fanbox 官方：標題
            for handle in self._safe(self.fanbox_creators, h.author):
                for post in self._safe(self.fanbox_posts, handle):
                    add(post, "fanbox", "fanbox")
            # 4. iwara：標題
            for u in self._safe(self.iwara_users, h.author):
                for v in self._safe(self.iwara_user_videos, u["id"]):
                    add({**v, "uid": u["id"]}, "iwara", "iwara")
            if m := self._pick(scored, ACCEPT):
                return m
        if not use_global:
            return None
        # 全站搜尋：只拿夠長的標題，且要幾乎一樣
        titles = sorted({t for t in h.titles if len(_core(t)) >= 6}, key=len, reverse=True)[:2]
        scored = []
        for t in titles:
            q = _core(t)
            try:
                for v in self.iwara_search(q):
                    s, why = self.score(v, h)
                    if h.author and similarity(h.author, v.get("author") or "") > 0.8:
                        s, why = s + 0.1, why + "＋作者同名"
                    scored.append((s, why, "iwara", "iwara", v))
            except NetError:
                pass
            for mirror in self.mirrors[:1]:
                try:
                    for post in self.mirror_search(mirror, q):
                        s, why = self.score(post, h)
                        scored.append((s, why, post["service"], mirror, post))
                except NetError:
                    pass
        return self._pick(scored, 0.9)
