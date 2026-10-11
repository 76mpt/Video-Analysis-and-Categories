"""Yandex 以圖搜圖（備援，預設關閉）：只對「不知作者、文字查詢都失敗」的影片，送一張畫面去查。

Yandex 沒有正式 API，這裡用 PicImageSearch 套件爬它的搜尋結果頁。
結果多半是轉貼站，只從網址裡挑出看得出來源的（iwara、fantia、fanbox、patreon、kemono…），
交給 Lookup 再查作者；其餘只當參考。
ascii2d 有 Cloudflare 驗證，第一次請求就被擋，所以不接。

啟用：config.yaml 的 reverse_search 底下加 `yandex: true`（預設 false）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from vsort.idm import URL_RULES

MIN_INTERVAL = 10.0  # 秒；實測結果見回報，被擋時會自動停用到 BLOCK_COOLDOWN 之後
BLOCK_COOLDOWN = 30 * 60


class YandexBlocked(Exception):
    pass


def sources_in(urls: list[str]) -> list[tuple[str, str]]:
    """從搜尋結果網址挑出 (網站, 貼文ID)。"""
    out = []
    for u in urls:
        for site, rx in URL_RULES:
            m = rx.search(u)
            if m:
                gd = m.groupdict()
                out.append(((gd.get("site") or site).lower(), gd["id"]))
                break
    return list(dict.fromkeys(out))


class YandexSearch:
    def __init__(self, cache_dir: Path, min_interval: float = MIN_INTERVAL):
        self.cache_dir = Path(cache_dir) / "yandex"
        self.min_interval = min_interval
        self._last = 0.0
        self._blocked_until = 0.0

    async def _search(self, image: Path) -> list[dict]:
        from PicImageSearch import Network, Yandex

        async with Network() as client:
            r = await Yandex(client=client).search(file=str(image))
        raw = r.raw or []
        if not raw and "captcha" in (str(getattr(r, "origin", "")) + str(getattr(r, "url", ""))).lower():
            raise YandexBlocked("Yandex 要求驗證碼")
        return [{"title": getattr(x, "title", "") or "", "url": getattr(x, "url", "") or ""} for x in raw[:30]]

    def search(self, image: Path, key: str) -> list[dict] | None:
        """回傳結果清單；被擋或冷卻中回傳 None（下次再試）。key 用來快取。"""
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        now = time.time()
        if now < self._blocked_until:
            return None
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            res = asyncio.run(self._search(image))
        except YandexBlocked:
            self._blocked_until = time.time() + BLOCK_COOLDOWN
            return None
        except Exception:  # 網路或頁面格式問題：當作這次查不到
            return None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res
