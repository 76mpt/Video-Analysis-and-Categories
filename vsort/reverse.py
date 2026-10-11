"""以圖搜圖找來源（F26，選用）：把一張關鍵幀送到 SauceNAO 反查原始作品頁面與作者。

預設關閉。要用的話在 secrets.yaml 填 saucenao.api_key（到 saucenao.com 免費註冊取得），
並在 config.yaml 把 reverse_search.enabled 設成 true。只有「不知道作者」的影片會送一張畫面出去，
結果只當建議，不會自動填入。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from vsort.net import Http, NetError

API = "https://saucenao.com/search.php"


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def parse_results(d: dict, min_similarity: float = 70.0) -> list[dict]:
    out = []
    for r in d.get("results") or []:
        h, data = r.get("header") or {}, r.get("data") or {}
        try:
            sim = float(h.get("similarity") or 0)
        except ValueError:
            sim = 0.0
        if sim < min_similarity:
            continue
        author = (data.get("member_name") or data.get("author_name") or data.get("creator")
                  or data.get("artist") or None)
        if isinstance(author, list):
            author = ", ".join(author)
        out.append({"similarity": sim, "index": h.get("index_name"), "author": author,
                    "title": data.get("title") or data.get("source"), "urls": data.get("ext_urls") or []})
    return out


class ReverseSearch:
    def __init__(self, api_key: str, http: Http | None = None, min_similarity: float = 70.0):
        self.key = api_key
        self.http = http or Http(min_interval=6.0)  # 免費額度約每 30 秒 4 次
        self.min_similarity = min_similarity

    def search(self, image: Path, cache: Path | None = None) -> list[dict]:
        if cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        body, ctype = _multipart("file", image.name, image.read_bytes())
        url = f"{API}?output_type=2&numres=5&api_key={self.key}"
        raw = self.http.request(url, headers={"Content-Type": ctype}, method="POST", raw_body=body)
        try:
            results = parse_results(json.loads(raw.decode("utf-8")), self.min_similarity)
        except ValueError:
            raise NetError("other", "SauceNAO 回應格式不對") from None
        if cache:
            cache.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        return results
