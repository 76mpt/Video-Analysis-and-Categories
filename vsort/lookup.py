"""來源網站查詢（F07）。

- iwara：官方 API；需要登入才看得到的影片，若 secrets.yaml 有帳號密碼就登入再查。
- fanbox、fantia、patreon：依你的決定，先查公開資訊 → 查不到再查 kemono 等第三方站 → 最後才用 cookie。
- 查詢結果快取在 Workspace\\_cache，試跑和實際執行不會重複連線。
- 連線暫時失敗時檔案留在 02_Parsed_Staging，下次 process 會再試。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from vsort.config import PROJECT_ROOT, Config
from vsort.dictionary import Dictionary, normalize
from vsort.net import Http, NetError
from vsort.parser import Parsed, Parser
from vsort.process import Enrichment

NOTFOUND_TTL = 7 * 86400


@dataclass
class SourceInfo:
    site: str
    post_id: str
    title: str | None = None
    author_name: str | None = None
    account_id: str | None = None
    account_handle: str | None = None
    tags: list[str] = field(default_factory=list)
    rating: str | None = None
    published: str | None = None
    via: str = "official"  # official / kemono / cookie / login


SECRET_KEYS = {"iwara": ("email", "password"), "cookies": ("fanbox", "fantia", "patreon"),
               "saucenao": ("api_key",)}
_warned: set[str] = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        import sys
        print(f"注意：{msg}", file=sys.stderr, flush=True)


def _repair_section(name: str, value) -> dict:
    """冒號後面少了空格時（例如 `api_key:abc`），YAML 會把整段讀成一行字；這裡把它拆回欄位。"""
    if isinstance(value, dict) or value is None:
        return value or {}
    keys = SECRET_KEYS.get(name, ())
    text = str(value)
    alt = "|".join(re.escape(k) for k in keys)
    out = {}
    for m in re.finditer(rf"(?:^|\s)({alt})\s*:\s*(.*?)(?=\s+(?:{alt})\s*:|$)", text, re.S):
        out[m.group(1)] = m.group(2).strip() or None
    _warn(f"secrets.yaml 的「{name}」格式不對：冒號後面要加一個空格（例如 `{keys[0] if keys else 'key'}: 值`）。"
          f"這次先自動拆開讀取{'成功' if out else '失敗'}，請修正檔案。")
    return out


def load_secrets(path: Path | None = None) -> dict:
    """讀 secrets.yaml。格式有誤時給清楚的提示，不讓程式當掉。"""
    path = path or PROJECT_ROOT / "secrets.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        _warn(f"secrets.yaml 讀不進來（{e.__class__.__name__}），先當作沒填。請檢查縮排和冒號後面的空格。")
        return {}
    if not isinstance(data, dict):
        _warn("secrets.yaml 格式不對（最外層要是 iwara:、cookies:、saucenao: 這幾段），先當作沒填。")
        return {}
    return {k: _repair_section(k, v) if k in SECRET_KEYS else v for k, v in data.items()}


def weak_title(t: str | None) -> bool:
    """檔名標題看不出內容（太短、像代碼）時，改用網站上的標題。"""
    if not t:
        return True
    s = t.strip()
    if len(s) <= 3:
        return True
    if re.fullmatch(r"[A-Za-z]{0,8}[ _\-]?\d{0,6}[A-Za-z]?", s):
        return True
    return s.lower() in {"untitled", "bonus", "test", "video", "movie", "sample", "full", "omake"}


class Lookup:
    def __init__(self, cfg: Config, secrets: dict | None = None, http: Http | None = None):
        lc = cfg.data.get("lookup") or {}
        self.enabled = lc.get("enabled", True)
        self.kemono = lc.get("kemono_domain", "kemono.cr")
        self.use_kemono = lc.get("use_kemono", True)
        self.secrets = secrets if secrets is not None else load_secrets()
        self.http = http or Http(min_interval=float(lc.get("min_interval_seconds", 1.0)))
        self.cache_dir = cfg.layout.workspace / "_cache"
        self._iwara_token: str | None = None
        self._iwara_login_failed = False
        self.mirrors = tuple(lc.get("mirrors") or (self.kemono, "pawchive.st"))
        self.use_idm = lc.get("use_idm", True)
        self.use_hints = lc.get("use_hints", True)
        self.use_oreno3d = lc.get("use_oreno3d", True)
        self._idm = None
        self._hints = None
        self._oreno = None
        # 每一站連續失敗（被擋、逾時）幾次就跳過，不卡住整批；這批跳過的影片留在 02 下次再試
        self.max_failures = int(lc.get("max_failures", 3))
        self._fails: dict[str, int] = {}

    def _blocked(self, site: str) -> bool:
        return self._fails.get(site, 0) >= self.max_failures

    def _record(self, site: str, ok: bool) -> None:
        if ok:
            self._fails[site] = 0
        else:
            self._fails[site] = self._fails.get(site, 0) + 1
            if self._fails[site] == self.max_failures:
                _warn(f"{site} 連續 {self.max_failures} 次查詢失敗（可能被擋），這一批先跳過，影片留在 02 下次再試。")

    # ---------- 沒有 ID 時找來源：IDM 下載紀錄 → 檔名線索 → oreno3D ----------

    @property
    def idm(self):
        if self._idm is None:
            try:
                from vsort.idm import IdmIndex
                self._idm = IdmIndex()
            except Exception as e:  # 不是 Windows、沒裝 IDM、登錄檔讀不到
                _warn(f"讀不到 IDM 下載紀錄（{e.__class__.__name__}），略過這一步。")
                self._idm = False
        return self._idm or None

    @property
    def hints(self):
        if self._hints is None:
            from vsort.hints import HintSearcher
            self._hints = HintSearcher(self.cache_dir, mirrors=self.mirrors)
        return self._hints

    @property
    def oreno(self):
        if self._oreno is None:
            from vsort.oreno3d import Oreno3D
            self._oreno = Oreno3D(self.cache_dir)
        return self._oreno

    def discover(self, p: Parsed, path: Path) -> dict | None:
        """檔名裡沒有網站 ID 時，依序用 IDM 下載紀錄、檔名線索搜尋、oreno3D 找出來源。"""
        if self.use_idm and self.idm is not None:
            try:
                m = self.idm.match(path)
            except OSError:
                m = None
            if m and m.site and m.source_id:
                return {"site": m.site, "post_id": m.source_id, "artist": m.account,
                        "found_by": f"IDM 下載紀錄（{m.how}）"}
        if self.use_hints and not self._blocked("hints"):
            try:
                from vsort.hints import extract_hints
                hm = self.hints.search(extract_hints(path, p))
                self._record("hints", True)
            except NetError as e:
                hm = None
                if e.kind == "retry":
                    self._record("hints", False)
            if hm:
                return {"site": hm.site, "post_id": hm.source_id, "artist": hm.account_id,
                        "found_by": f"檔名線索：{hm.reason}（{hm.score:.2f}）"}
        mv = self.oreno_find(p)
        if mv and mv.iwara_id:
            return {"site": "iwara", "post_id": mv.iwara_id, "artist": None, "found_by": "oreno3D 標題搜尋"}
        return None

    def oreno_find(self, p: Parsed):
        if not self.use_oreno3d or self._blocked("oreno3d"):
            return None
        try:
            r = self.oreno.find([t for t in [p.title, *p.folder_hints] if t], p.author)
            self._record("oreno3d", True)
        except NetError as e:
            if e.kind == "retry":
                self._record("oreno3d", False)
            return None
        return r[0] if r else None

    def oreno_info(self, mv, pid: str) -> SourceInfo:
        """iwara 已刪除的影片：用 oreno3D 保存的作者、原作、角色、標籤補上。"""
        return SourceInfo(site="iwara", post_id=pid, title=mv.title, author_name=mv.author,
                          tags=[*mv.tags, *mv.origins, *mv.characters], published=mv.published, via="oreno3d")

    # ---------- 快取 ----------

    def _cache_path(self, site: str, pid: str) -> Path:
        return self.cache_dir / site / f"{re.sub(r'[^A-Za-z0-9_-]', '_', pid)}.json"

    def _cache_get(self, site: str, pid: str):
        p = self._cache_path(site, pid)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if d.get("status") == "notfound":
            if time.time() - d.get("ts", 0) > NOTFOUND_TTL:
                return None
            return NetError("notfound", "（快取）查無此作品")
        return SourceInfo(**d["info"])

    def _cache_put(self, site: str, pid: str, info: SourceInfo | None) -> None:
        p = self._cache_path(site, pid)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {"status": "ok", "info": asdict(info)} if info else {"status": "notfound"}
        d["ts"] = time.time()
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    # ---------- 對外 ----------

    def fetch(self, site: str, pid: str, artist: str | None = None) -> SourceInfo:
        cached = self._cache_get(site, pid)
        if isinstance(cached, NetError):
            raise cached
        if cached:
            return cached
        if self._blocked(site):
            raise NetError("retry", f"{site} 這一批連續失敗，先跳過")
        try:
            info = self._iwara(pid) if site == "iwara" else self._paid(site, pid, artist)
        except NetError as e:
            if e.kind == "notfound":
                self._cache_put(site, pid, None)
            self._record(site, e.kind != "retry")
            raise
        self._record(site, True)
        self._cache_put(site, pid, info)
        return info

    def _paid(self, site: str, pid: str, artist: str | None = None) -> SourceInfo:
        official = {"fanbox": self._fanbox, "fantia": self._fantia, "patreon": self._patreon}[site]
        errors = []
        try:
            return official(pid)
        except NetError as e:
            if e.kind == "retry":
                raise
            errors.append(e)
        if self.use_kemono:
            try:
                from vsort.mirrors import fetch_from_mirrors  # kemono 查不到再查 pawchive
                return fetch_from_mirrors(self.http, site, pid, artist, mirrors=self.mirrors)
            except NetError as e:
                if e.kind == "retry":
                    raise
                errors.append(e)
        cookie = (self.secrets.get("cookies") or {}).get(site)
        if cookie:
            info = official(pid, cookie=cookie)
            info.via = "cookie"
            return info
        raise errors[0] if errors[0].kind == "auth" else NetError("notfound", "公開資訊與第三方站都查不到")

    # ---------- iwara ----------

    def _iwara_login(self) -> str | None:
        acc = self.secrets.get("iwara") or {}
        if self._iwara_token or self._iwara_login_failed or not acc.get("email") or not acc.get("password"):
            return self._iwara_token
        try:
            d = self.http.json("https://api.iwara.tv/user/login", method="POST",
                               data={"email": acc["email"], "password": acc["password"]})
            token = d.get("token")
            try:
                d2 = self.http.json("https://api.iwara.tv/user/token", method="POST",
                                    headers={"Authorization": f"Bearer {token}"})
                token = d2.get("accessToken") or token
            except NetError:
                pass
            self._iwara_token = token
        except NetError:
            self._iwara_login_failed = True
        return self._iwara_token

    def _iwara(self, vid: str) -> SourceInfo:
        url = f"https://api.iwara.tv/video/{vid}"
        via = "official"
        try:
            d = self.http.json(url, headers={"Authorization": f"Bearer {self._iwara_token}"}
                               if self._iwara_token else None)
        except NetError as e:
            if e.kind not in ("auth", "notfound") or not self._iwara_login():
                raise
            d = self.http.json(url, headers={"Authorization": f"Bearer {self._iwara_token}"})
            via = "login"
        user = d.get("user") or {}
        return SourceInfo(site="iwara", post_id=vid, title=d.get("title"), author_name=user.get("name"),
                          account_id=user.get("id"), account_handle=user.get("username"),
                          tags=[t.get("id") for t in d.get("tags") or [] if t.get("id")],
                          rating=d.get("rating"), published=d.get("createdAt"), via=via)

    # ---------- 付費站官方公開資訊 ----------

    def _fanbox(self, pid: str, cookie: str | None = None) -> SourceInfo:
        h = {"Origin": "https://www.fanbox.cc", "Referer": "https://www.fanbox.cc/"}
        if cookie:
            h["Cookie"] = cookie
        d = self.http.json(f"https://api.fanbox.cc/post.info?postId={pid}", headers=h)
        post = (d.get("body") or {}).get("post") or d.get("body") or {}
        if not post.get("title"):
            raise NetError("notfound", "fanbox 沒有回傳標題")
        user = post.get("user") or {}
        return SourceInfo(site="fanbox", post_id=pid, title=post.get("title"), author_name=user.get("name"),
                          account_id=str(user.get("userId") or "") or None,
                          account_handle=post.get("creatorId"), tags=list(post.get("tags") or []),
                          published=post.get("publishedDatetime"))

    def _fantia(self, pid: str, cookie: str | None = None) -> SourceInfo:
        h = {"Cookie": cookie} if cookie else {}
        html = self.http.text(f"https://fantia.jp/posts/{pid}", headers=dict(h))
        m = re.search(r'name="csrf-token" content="([^"]+)"', html)
        if not m:
            raise NetError("other", "fantia 頁面格式變了")
        h.update({"X-CSRF-Token": m.group(1), "X-Requested-With": "XMLHttpRequest",
                  "Referer": f"https://fantia.jp/posts/{pid}"})
        try:
            d = self.http.json(f"https://fantia.jp/api/v1/posts/{pid}", headers=h)
        except NetError as e:
            if e.kind == "other":  # 不存在的貼文會回 422
                raise NetError("notfound", str(e)) from None
            raise
        post = d.get("post") or {}
        fc = post.get("fanclub") or {}
        return SourceInfo(site="fantia", post_id=pid, title=post.get("title"),
                          author_name=fc.get("creator_name") or fc.get("name"),
                          account_id=str(fc.get("id") or "") or None, account_handle=fc.get("name"),
                          tags=[t.get("name") for t in post.get("tags") or [] if t.get("name")],
                          rating=post.get("rating"), published=post.get("posted_at"))

    def _patreon(self, pid: str, cookie: str | None = None) -> SourceInfo:
        h = {"Cookie": cookie} if cookie else {}
        url = (f"https://www.patreon.com/api/posts/{pid}?include=user,campaign"
               "&fields[user]=full_name,vanity&fields[campaign]=name,vanity&fields[post]=title,published_at")
        d = self.http.json(url, headers=h)
        data = d.get("data") or {}
        attrs = data.get("attributes") or {}
        rel = data.get("relationships") or {}
        user_id = ((rel.get("user") or {}).get("data") or {}).get("id")
        names = {(i.get("type"), i.get("id")): i.get("attributes") or {} for i in d.get("included") or []}
        camp_id = ((rel.get("campaign") or {}).get("data") or {}).get("id")
        camp = names.get(("campaign", camp_id), {})
        user = names.get(("user", user_id), {})
        if not attrs.get("title"):
            raise NetError("notfound", "patreon 沒有回傳標題")
        return SourceInfo(site="patreon", post_id=pid, title=attrs.get("title"),
                          author_name=camp.get("name") or user.get("full_name"), account_id=user_id,
                          account_handle=camp.get("vanity") or user.get("vanity"),
                          published=attrs.get("published_at"))

    # ---------- kemono（第三方備援）----------

    def _kemono_post(self, site: str, pid: str) -> SourceInfo:
        base = f"https://{self.kemono}/api/v1"
        h = {"Accept": "text/css"}  # kemono 的 API 要求這個標頭
        ref = self.http.json(f"{base}/{site}/post/{pid}", headers=h)
        artist = ref.get("artist_id")
        if not artist:
            raise NetError("notfound", "kemono 查無此貼文")
        d = self.http.json(f"{base}/{site}/user/{artist}/post/{pid}", headers=h)
        post = d.get("post", d)
        name = None
        try:
            name = self.http.json(f"{base}/{site}/user/{artist}/profile", headers=h).get("name")
        except NetError:
            pass
        return SourceInfo(site=site, post_id=pid, title=post.get("title"), author_name=name,
                          account_id=str(artist), tags=list(post.get("tags") or []),
                          published=post.get("published"), via="kemono")


# ---------- 接進 process 的 enricher ----------

ANIM_TAGS = {"mmd": "MMD", "koikatsu": "Koikatsu", "vam": "VAM", "virt a mate": "VAM", "3d": "3D",
             "blender": "3D", "honey select": "3D",
             "sfm": "3D", "source filmmaker": "3D", "vam": "3D", "2d": "2D", "live2d": "2D"}


def candidates(p: Parsed) -> list[tuple[str, str]]:
    out = []
    if p.site and p.source_id:
        out.append((p.site, p.source_id))
    if p.site_guess and p.code and not out:
        out.append((p.site_guess, p.code))
    return out


def make_enricher(lookup: Lookup, dic: Dictionary):
    def enrich(p: Parsed, data: dict, path: Path | None = None) -> Enrichment | None:
        if not lookup.enabled:
            return None
        cands = candidates(p)
        artist, found_by = None, None
        if not cands and path is not None:
            found = lookup.discover(p, Path(path))
            if found:
                cands = [(found["site"], found["post_id"])]
                artist, found_by = found.get("artist"), found["found_by"]
        if not cands:
            return None
        site, pid = cands[0]
        rec: dict = {"site": site, "post_id": pid, "t": datetime.now().isoformat(timespec="seconds")}
        if found_by:
            rec["found_by"] = found_by
        err: NetError | None = None
        try:
            info = lookup.fetch(site, pid, artist)
        except NetError as exc:
            err, info = exc, None
            mv = lookup.oreno_find(p) if site == "iwara" and exc.kind == "notfound" else None
            if mv and (not mv.iwara_id or mv.iwara_id == pid):
                info = lookup.oreno_info(mv, pid)  # iwara 刪片：改用 oreno3D 保存的資料
        if info is None:
            e = err
            rec.update(status=e.kind, error=str(e))
            if e.kind == "retry":
                return Enrichment(data={"lookup": rec}, retry=True, note=f"查詢 {site} 暫時失敗，留在 02 下次再試")
            label = {"notfound": "查無此作品（可能已刪除）", "auth": "需要登入才看得到"}.get(e.kind, str(e))
            return Enrichment(data={"lookup": rec}, note=f"{site}：{label}")

        rec.update(status="ok", **asdict(info))
        out: dict = {}
        tags_text = " ".join(info.tags or [])
        haystack = " ".join(x for x in [info.title or "", tags_text] if x)
        works = dic.match_works(haystack)
        chars = dic.match_characters(haystack, works)
        out["works"] = works + [w for _, w in chars if w and w not in works]
        out["characters"] = [c for c, _ in chars]
        labels = [ANIM_TAGS[normalize(t)] for t in info.tags or [] if normalize(t) in ANIM_TAGS]
        if labels:
            out["anim_type"] = list(dict.fromkeys(labels))  # 例如 mmd＋blender → MMD,3D
        if weak_title(p.title) and info.title:
            out["title"] = info.title

        # 作者：帳號 ID 對到別名表 → 用標準名；否則沿用你的資料夾名稱，並把對應關係列入待確認
        canonical = dic.author_by_account(info.site, info.account_id) if info.account_id else None
        if canonical:
            out["author"] = canonical
        elif info.author_name or info.account_id:
            proposed = p.author or info.author_name
            out["author"] = proposed
            rec["author_candidate"] = {"canonical": proposed, "site": info.site,
                                       "account_id": info.account_id, "name": info.author_name,
                                       "handle": info.account_handle}
        via = {"kemono": "（kemono）", "pawchive": "（pawchive）", "oreno3d": "（oreno3D，iwara 已刪除）"}.get(info.via, "")
        note = (f"{found_by} → " if found_by else "") + \
            f"{site} 查到：{info.author_name or '?'}／{info.title or '?'}{via}"
        return Enrichment(info=out, data={"lookup": rec}, note=note)

    return enrich


def build_enrichers(cfg: Config, parser: Parser) -> list:
    lookup = Lookup(cfg)
    if not lookup.enabled:
        return []
    return [make_enricher(lookup, parser.dic)]
