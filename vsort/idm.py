"""從 IDM（Internet Download Manager）的下載紀錄找出影片的來源網址。

IDM 把每筆下載存在登錄檔 HKCU\\Software\\DownloadManager\\<編號>：
- Url0：實際下載網址（常帶有原始檔名，例如 gofile、kemono 的 ?f=）
- owWPage／Referer：按下載時所在的網頁（iwara 影片頁、fantia 貼文頁、kemono 貼文頁…）
- FileName／FR_FNCD：網址上的檔名、伺服器給的檔名
- FileSize：檔案大小（位元組），改過檔名也能用大小對回來

只讀取，不寫入；cookie、密碼、標頭等欄位一律不讀。
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REG_PATH = r"Software\DownloadManager"
# 只讀這些欄位（cookie、密碼、自訂標頭不碰）
FIELDS = ("Url0", "owWPage", "Referer", "FileName", "FR_FNCD", "FileSize", "Downloaded", "dateAdded")
MIN_SIZE = 1_000_000  # 太小的檔案（圖片、文字）大小容易撞號，不拿來比

# (網站, 正規式)；ID 群組叫 id，作者帳號群組叫 account
URL_RULES = [
    ("iwara", re.compile(r"iwara\.tv/videos?/(?P<id>[A-Za-z0-9]{8,20})", re.I)),
    ("fantia", re.compile(r"fantia\.jp/posts/(?P<id>\d{4,})", re.I)),
    ("fanbox", re.compile(r"downloads\.fanbox\.cc/files/post/(?P<id>\d{4,})/", re.I)),
    ("fanbox", re.compile(r"(?:(?P<account>[\w\-]+)\.)?fanbox\.cc/(?:@[\w\-]+/)?posts/(?P<id>\d{4,})", re.I)),
    ("patreon", re.compile(r"patreon\.com/posts/(?:[\w\-]*?-)?(?P<id>\d{6,})", re.I)),
    # kemono 類鏡像站：/<站>/user/<帳號>/post/<貼文>
    ("mirror", re.compile(r"(?:kemono|coomer|pawchive)\.\w+/(?P<site>fanbox|fantia|patreon)/user/"
                          r"(?P<account>[\w\-]+)/post/(?P<id>\d+)", re.I)),
]
# 只告訴我們「從哪個網站下載」、但查不到作者的網站
FILE_HOSTS = ("pixeldrain", "gofile", "mega.nz", "drive.google", "dropbox", "transfer.it", "mediafire")


@dataclass
class IdmEntry:
    key: str
    url: str
    page: str
    referer: str
    filename: str          # 網址或伺服器給的檔名（已解碼）
    size: int | None
    date: str

    def urls(self) -> list[str]:
        return [u for u in (self.page, self.referer, self.url) if u]


@dataclass
class IdmMatch:
    entry: IdmEntry
    how: str                       # size+name / size / name
    site: str | None = None        # iwara / fantia / fanbox / patreon
    source_id: str | None = None
    account: str | None = None     # fanbox 子網域、kemono 的帳號 ID
    host: str | None = None        # 下載頁網域（例如 pixeldrain.com）
    orig_name: str | None = None   # 下載網址帶的原始檔名，可再丟給 Parser 解析

    @property
    def page_url(self) -> str:
        return self.entry.page or self.entry.referer or self.entry.url


def _qword(v) -> int | None:
    if isinstance(v, (bytes, bytearray)) and len(v) >= 8:
        return int.from_bytes(v[:8], "little")
    if isinstance(v, int):
        return v
    return None


def _clean_name(n: str) -> str:
    return unquote(n.split("?")[0]).strip()


def read_registry() -> list[IdmEntry]:
    """讀 IDM 在登錄檔裡的下載清單（唯讀）。不是 Windows 或沒裝 IDM 就回傳空清單。"""
    if sys.platform != "win32":
        return []
    import winreg

    try:
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_READ)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            name = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not name.isdigit():
            continue
        vals = {}
        try:
            with winreg.OpenKey(root, name, 0, winreg.KEY_READ) as k:
                for f in FIELDS:
                    try:
                        vals[f] = winreg.QueryValueEx(k, f)[0]
                    except OSError:
                        pass
        except OSError:
            continue
        out.append(entry_from_values(name, vals))
    return out


def entry_from_values(key: str, vals: dict) -> IdmEntry:
    size = _qword(vals.get("FileSize")) or _qword(vals.get("Downloaded"))
    fname = vals.get("FR_FNCD") or vals.get("FileName") or ""
    return IdmEntry(key=key, url=vals.get("Url0") or "", page=vals.get("owWPage") or "",
                    referer=vals.get("Referer") or "", filename=_clean_name(str(fname)),
                    size=size, date=vals.get("dateAdded") or "")


def source_of(e: IdmEntry) -> dict:
    """從一筆下載紀錄的網址找出 (網站, 貼文ID, 帳號)。"""
    res: dict = {}
    for u in e.urls():
        for site, rx in URL_RULES:
            m = rx.search(u)
            if not m:
                continue
            gd = m.groupdict()
            res = {"site": (gd.get("site") or site).lower(), "source_id": gd["id"],
                   "account": gd.get("account")}
            if res["account"] and res["account"].lower() in ("www", "api", "downloads"):
                res["account"] = None
            break
        if res:
            break
    host = urlparse(e.page or e.referer or e.url).netloc.lower()
    res["host"] = host or None
    # 原始檔名：gofile 網址最後一段、kemono 的 ?f=、iwara 的 download=
    pu = urlparse(e.url)
    q = parse_qs(pu.query)
    orig = (q.get("f") or q.get("download") or [None])[0]
    if not orig and any(h in pu.netloc for h in ("gofile",)):
        orig = unquote(pu.path.rsplit("/", 1)[-1])
    if orig:
        res["orig_name"] = orig
    return res


class IdmIndex:
    """用檔案大小與檔名把影片對回 IDM 下載紀錄。"""

    def __init__(self, entries: list[IdmEntry] | None = None):
        self.entries = read_registry() if entries is None else entries
        self.by_size: dict[int, list[IdmEntry]] = defaultdict(list)
        self.by_name: dict[str, list[IdmEntry]] = defaultdict(list)
        for e in self.entries:
            if e.size and e.size >= MIN_SIZE:
                self.by_size[e.size].append(e)
            for n in {e.filename, _clean_name(e.url.rsplit("/", 1)[-1])}:
                if n:
                    self.by_name[n.lower()].append(e)

    def __len__(self) -> int:
        return len(self.entries)

    def match(self, path: str | os.PathLike, size: int | None = None) -> IdmMatch | None:
        p = Path(path)
        if size is None:
            try:
                size = p.stat().st_size
            except OSError:
                size = None
        by_size = self.by_size.get(size, []) if size else []
        by_name = self.by_name.get(p.name.lower(), [])
        both = [e for e in by_size if e in by_name]
        if both:
            e, how = both[-1], "size+name"
        elif len(by_size) == 1 or (by_size and len({self._src_key(x) for x in by_size}) == 1):
            e, how = by_size[-1], "size"   # 大小獨一無二，或撞號的幾筆其實是同一個來源
        elif by_name and not size:
            e, how = by_name[-1], "name"
        elif len(by_name) == 1 and not by_size:
            # 名稱對到但大小不同：可能重新轉檔或沒下載完，只在名稱夠長時採用
            if len(p.stem) < 12:
                return None
            e, how = by_name[0], "name"
        else:
            return None
        return IdmMatch(entry=e, how=how, **source_of(e))

    @staticmethod
    def _src_key(e: IdmEntry):
        s = source_of(e)
        return (s.get("site"), s.get("source_id"))
