"""作者先驗：同一位作者的作品類別與動畫類型多半一致（標註資料模擬：類別 89%、動畫類型 93%）。

資料來源（都不用額外標註）：
- 你在標註網頁標過的影片（dict/style_labels.csv，作者取自資料夾名稱）
- 04 收藏庫裡已歸檔的影片（自動確定與人工確認的都算，檔名裡就有作者、動畫類型、類別）

用法：
- 類別：把分類器的機率乘上這位作者過去的類別比例（平滑後）的 0.5 次方，再正規化。251 部留一法模擬
  （分類器含聲音特徵、門檻 0.7）：放行 80% → 87%，放行的正確率維持 90%；次方 1.0 再多 1%，但正確率掉到 89%。
- 動畫類型：同作者 ≥3 部且 ≥80% 一致就直接沿用（原本只看收藏庫，而收藏庫一開始是空的，所以從沒生效）。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from vsort.config import Config
from vsort.dictionary import normalize

CATS = ["Dance", "Mix", "Scene"]


class AuthorPriors:
    def __init__(self, cfg: Config, min_videos: int = 2, smoothing: float = 1.0, weight: float = 0.5,
                 style_min: int = 3, style_agree: float = 0.8):
        self.min_videos, self.smoothing, self.weight = min_videos, smoothing, weight
        self.style_min, self.style_agree = style_min, style_agree
        self.cat: dict[str, Counter] = defaultdict(Counter)
        self.style: dict[str, Counter] = defaultdict(Counter)
        self.names: dict[str, str] = {}
        self.sources: dict[str, set[str]] = defaultdict(set)  # 作者 → 已算進來的影片 key，避免重複
        self._load_library(cfg)
        self._load_labels(cfg)
        self.global_cat = Counter()
        for c in self.cat.values():
            self.global_cat.update(c)

    def _key(self, author: str) -> str:
        return normalize(author or "")

    def add(self, author: str, category: str | None, anim: list[str] | None, key: str) -> None:
        k = self._key(author)
        if not k or key in self.sources[k]:
            return
        self.sources[k].add(key)
        self.names.setdefault(k, author)
        if category in CATS:
            self.cat[k][category] += 1
        if anim:
            self.style[k][",".join(anim)] += 1

    def _load_library(self, cfg: Config) -> None:
        from vsort.naming import parse_final_name
        from vsort.process import _videos
        for v in _videos(cfg.layout.library, cfg):
            f = parse_final_name(v.name)
            if f and f.get("author"):
                self.add(f["author"], f.get("category"), f.get("anim_type"), f"lib:{v}")

    def _load_labels(self, cfg: Config) -> None:
        try:
            from vsort.labeling import load_labels
            from vsort.process import make_parser
            parser = make_parser(cfg)
            for key, lab in load_labels(cfg).items():
                p = Path(lab["path"])
                a = parser.parse(p.name, list(p.parent.parts[1:])).author  # 去掉磁碟代號；通用資料夾名稱會被跳過
                if a:
                    self.add(a, lab.get("category") or None, lab.get("labels") or None, f"label:{key}")
        except Exception:
            pass

    # ---------- 類別 ----------

    def category_prior(self, author: str | None, exclude_key: str | None = None) -> dict[str, float] | None:
        """這位作者過去的類別比例（平滑）。資料不足回傳 None。"""
        k = self._key(author or "")
        cnt = Counter(self.cat.get(k) or {})
        n = sum(cnt.values())
        if n < self.min_videos:
            return None
        total = sum(self.global_cat.values()) or 1
        return {c: (cnt[c] + self.smoothing * self.global_cat[c] / total) / (n + self.smoothing) for c in CATS}

    def apply_category(self, probs: dict[str, float], author: str | None) -> tuple[dict[str, float], dict | None]:
        """把分類器機率乘上作者先驗（權重 weight 次方）再正規化。回傳 (新機率, 先驗或 None)。"""
        prior = self.category_prior(author)
        if not prior or not probs:
            return probs, None
        q = {c: probs.get(c, 0.0) * (prior[c] ** self.weight) for c in probs}
        s = sum(q.values())
        if s <= 0:
            return probs, None
        n = sum(self.cat[self._key(author or "")].values())
        return {c: v / s for c, v in q.items()}, {"n": n, **{c: round(prior[c], 3) for c in CATS}}

    # ---------- 動畫類型 ----------

    def anim_type(self, author: str | None) -> list[str] | None:
        k = self._key(author or "")
        c = self.style.get(k)
        if not c:
            return None
        label, n = c.most_common(1)[0]
        total = sum(c.values())
        if total >= self.style_min and n / total >= self.style_agree:
            return label.split(",")
        return None

    def styles(self) -> dict[str, list[str]]:
        """相容舊介面：{作者標準名: 動畫類型}。"""
        out = {}
        for k in self.style:
            a = self.anim_type(self.names.get(k, k))
            if a:
                out[self.names.get(k, k)] = a
        return out
