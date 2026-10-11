"""第三方鏡像站（kemono、pawchive）查單篇貼文。

兩站 API 格式相同（/api/v1/<站>/user/<帳號>/post/<貼文>）。差別：
- kemono 可以只用貼文 ID 反查帳號（/api/v1/<站>/post/<貼文>）。
- pawchive 沒有這個反查，要先知道帳號 ID（來自 IDM 的下載網址或檔名線索搜尋）。

Lookup._paid 目前是「官方公開資訊 → kemono → cookie」；接法見 fetch_from_mirrors 的說明。
"""

from __future__ import annotations

from vsort.lookup import SourceInfo
from vsort.net import Http, NetError

DEFAULT_MIRRORS = ("kemono.cr", "pawchive.st")
H = {"Accept": "text/css"}  # 兩站的 API 都要求這個標頭


def mirror_post(http: Http, mirror: str, site: str, pid: str, artist: str | None = None) -> SourceInfo:
    base = f"https://{mirror}/api/v1"
    if not artist:
        try:
            artist = http.json(f"{base}/{site}/post/{pid}", headers=H).get("artist_id")
        except NetError as e:
            if e.kind != "notfound":
                raise
        if not artist:
            raise NetError("notfound", f"{mirror} 無法只用貼文 ID 查（需要帳號 ID）")
    d = http.json(f"{base}/{site}/user/{artist}/post/{pid}", headers=H)
    post = d.get("post", d)
    if not post.get("title") and not post.get("id"):
        raise NetError("notfound", f"{mirror} 查無此貼文")
    name = None
    try:
        name = http.json(f"{base}/{site}/user/{artist}/profile", headers=H).get("name")
    except NetError:
        pass
    return SourceInfo(site=site, post_id=pid, title=post.get("title"), author_name=name,
                      account_id=str(artist), tags=list(post.get("tags") or []),
                      published=post.get("published"), via=mirror.split(".")[0])


def fetch_from_mirrors(http: Http, site: str, pid: str, artist: str | None = None,
                       mirrors=DEFAULT_MIRRORS) -> SourceInfo:
    """依序查各鏡像站，第一個查到的就回傳。

    接進 Lookup._paid：把原本 `self._kemono_post(site, pid)` 換成
    `fetch_from_mirrors(self.http, site, pid, artist, mirrors=(self.kemono, "pawchive.st"))`，
    artist 從 IDM 對應（IdmMatch.account）或檔名線索（HintMatch.account_id）帶進來。
    """
    errors: list[NetError] = []
    for m in mirrors:
        try:
            return mirror_post(http, m, site, pid, artist)
        except NetError as e:
            if e.kind == "retry" and not errors:
                errors.append(e)
                continue
            errors.append(e)
    retry = next((e for e in errors if e.kind == "retry"), None)
    raise retry or NetError("notfound", "；".join(str(e) for e in errors) or "鏡像站都查不到")
