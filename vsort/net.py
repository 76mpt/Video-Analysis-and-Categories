"""簡單的 HTTP 工具：逾時、gzip、每個網站的請求間隔、錯誤分類。只用標準函式庫。"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from urllib.parse import urlparse

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


class NetError(Exception):
    """kind：notfound（不存在／已刪除）、auth（需要登入或沒權限）、retry（暫時失敗，之後再試）、other。"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Http:
    def __init__(self, min_interval: float = 1.0, timeout: float = 20.0):
        self.min_interval = min_interval
        self.timeout = timeout
        self._last: dict[str, float] = {}
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            delay = self.min_interval - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
        self._last[host] = time.monotonic()

    def request(self, url: str, headers: dict | None = None, data: dict | None = None,
                method: str | None = None, raw_body: bytes | None = None) -> bytes:
        h = {"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"}
        h.update(headers or {})
        body = raw_body
        if data is not None:
            body = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
        self._wait(urlparse(url).netloc)
        req = urllib.request.Request(url, data=body, headers=h, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                raise NetError("notfound", f"HTTP {e.code}") from None
            if e.code in (401, 403):
                raise NetError("auth", f"HTTP {e.code}") from None
            if e.code == 429 or e.code >= 500:
                raise NetError("retry", f"HTTP {e.code}") from None
            raise NetError("other", f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise NetError("retry", f"連線失敗：{e}") from None

    def json(self, url: str, **kw) -> dict:
        raw = self.request(url, **kw)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise NetError("other", "回應不是 JSON") from None

    def text(self, url: str, **kw) -> str:
        kw.setdefault("headers", {})
        kw["headers"].setdefault("Accept", "text/html")
        return self.request(url, **kw).decode("utf-8", "replace")
