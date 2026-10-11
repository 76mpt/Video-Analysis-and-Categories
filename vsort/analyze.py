"""第二階段：AI 視覺分析與最終判定（F19 角色、F21 類別、F28 動畫類型、F22 補類別、F23 合併、F24 人工確認）。

流程：03 的影片 → 取關鍵幀 → 標籤模型 → 彙整成角色／類別／動畫類型與信心
→ 和檔名解析、網路查詢的結果合併 → 「已確定」進 04 收藏庫（依作者分資料夾），
「可能誤判」進 05_Review，附判斷理由與預覽圖。
AI 結果快取在 Workspace\\_cache\\ai，試跑後實際執行不會重算。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from vsort import sidecar as sc
from vsort.config import Config
from vsort.dictionary import Dictionary, normalize
from vsort.journal import Op, Plan
from vsort.naming import build_final_name, parse_final_name, sanitize_dirname, unique_path
from vsort.process import _videos



def load_tag_list(name: str) -> list[str]:
    """Tagger labels, one per line, from dict/<name>.txt (local, git-ignored) or dict/<name>.example.txt."""
    folder = Path(__file__).resolve().parent.parent / "dict"
    for fn in (f"{name}.txt", f"{name}.example.txt"):
        f = folder / fn
        if f.exists():
            return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
    return []


SCENE_TAGS = load_tag_list("scene_tags")  # frames scoring >= scene_frame_threshold on any of these count as Scene
UNKNOWN_AUTHOR_DIR = "作者未知"  # 收藏庫裡不知道作者的影片放這裡，之後另外處理
DANCE_TAGS = ["dancing", "idol", "microphone", "stage", "concert"]

DEFAULTS = {
    "frames": 16,
    "scene_frame_threshold": 0.45,   # 單張畫面算「場景」的門檻
    "dance_max_ratio": 0.10,       # 場景畫面比例 ≤ 這個值 → Dance
    "scene_min_ratio": 0.70,         # ≥ 這個值 → Scene；中間 → Mix
    "border_margin": 0.03,         # 比例離門檻這麼近就標「可能誤判」
    "character_threshold": 0.50,   # 角色出現在畫面的門檻
    "character_sure": 0.85,        # 高於這個值才算確定
    "character_min_frames": 2,
    "style_sure": 0.70,            # 零樣本：最高的類型機率高於這個值才算確定
    "style_label_threshold": 0.5,  # 分類器：每個類型機率 ≥ 這個值就標上（可複選）
    "style_label_thresholds": {"MMD": 0.3, "Koikatsu": 0.4},  # 個別類型的門檻：MMD 寧可多標
    "style_label_margin": 0.3,     # 分類器：每個機率都離門檻這麼遠才算確定（MMD 多標不算不確定）
    "style_label_floor": 0.1,      # 分類器：機率 ≤ 這個值就當作「明顯沒有」，不再要求離門檻夠遠（350 部交叉驗證：有把握 55% → 66%）
    "category_model": "combined",  # 類別由誰判斷：combined（畫面標籤＋Qwen3-VL，用你標的類別訓練）、qwen3vl 或 tags
    "category_sure_prob": 0.8,     # combined：最有把握的類別機率 ≥ 這個值才算確定
    "category_second_stage": True,  # 信心不足時才加跑備援版逐張判斷（每部多約 10 秒）
    "author_prior_weight": 0.5,    # 作者先驗：類別機率乘上同作者過去類別比例的這個次方（0 = 不用）
    "model": "pixai-labs/pixai-tagger-v1.0",
    "describe_model": "none",      # 描述模型（選用）：none、qwen3vl、joycaption；只用在「可能誤判」的影片
    "style_model": "openai/clip-vit-large-patch14",
}


@dataclass
class AIResult:
    frames: int = 0
    scene_ratio: float = 0.0
    dance_ratio: float = 0.0
    explicit_ratio: float = 0.0
    category: str | None = None
    category_sure: bool = False
    category_tags: str | None = None  # 畫面標籤比例判斷的類別
    category_vlm: str | None = None   # Qwen3-VL 判斷的類別
    category_model: str = ""
    description: str = ""             # Qwen3-VL 的文字描述
    category_probs: dict = field(default_factory=dict)  # combined 分類器的各類別機率
    category_prior: dict | None = None  # 作者先驗（套用時記錄：n 與各類別比例）
    timings: dict = field(default_factory=dict)  # 各步驟耗時（秒）
    characters: list[dict] = field(default_factory=list)  # {tag, prob, frames}
    character_candidates: list[dict] = field(default_factory=list)  # 人工確認用：前 5 名（含信心較低的）
    top_tags: list[str] = field(default_factory=list)
    style: dict = field(default_factory=dict)  # 動畫類型機率 {MMD, 3D, 2D}
    style_kind: str = "zeroshot"
    style_model: str = ""  # zeroshot：三選一的機率；classifier：每個類型各自的機率（可複選）
    preview: str | None = None
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(per_frame: list[dict[str, float]], char_tags: set[str], opts: dict) -> AIResult:
    """把每張畫面的標籤機率彙整成整部影片的判斷。純函式，方便測試與調整門檻。"""
    n = len(per_frame)
    r = AIResult(frames=n)
    if n == 0:
        return r
    thr = opts["scene_frame_threshold"]
    scene_frames = sum(1 for f in per_frame if max((f.get(t, 0) for t in SCENE_TAGS), default=0) >= thr)
    r.scene_ratio = scene_frames / n
    r.dance_ratio = sum(1 for f in per_frame if max(f.get(t, 0) for t in DANCE_TAGS) >= 0.35) / n
    r.explicit_ratio = sum(1 for f in per_frame if f.get("explicit", 0) >= 0.5) / n

    lo, hi, margin = opts["dance_max_ratio"], opts["scene_min_ratio"], opts["border_margin"]
    if r.scene_ratio <= lo:
        r.category = "Dance"
    elif r.scene_ratio >= hi:
        r.category = "Scene"
    else:
        r.category = "Mix"
    r.category_sure = n >= 4 and abs(r.scene_ratio - lo) > margin and abs(r.scene_ratio - hi) > margin

    agg: dict[str, list[float]] = {}
    for f in per_frame:
        for tag, p in f.items():
            if tag in char_tags and p >= opts["character_threshold"]:
                agg.setdefault(tag, []).append(p)
    chars = [{"tag": t, "prob": round(max(ps), 3), "frames": len(ps)}
             for t, ps in agg.items() if len(ps) >= min(opts["character_min_frames"], n)]
    r.characters = sorted(chars, key=lambda c: (-c["frames"], -c["prob"]))[:4]
    best: dict[str, list[float]] = {}
    for f in per_frame:
        for tag, p in f.items():
            if tag in char_tags and p >= 0.1:
                best.setdefault(tag, []).append(p)
    r.character_candidates = sorted(({"tag": t, "prob": round(max(ps), 3), "frames": len(ps)} for t, ps in best.items()),
                                    key=lambda c: -c["prob"])[:5]

    totals: dict[str, float] = {}
    for f in per_frame:
        for tag, p in f.items():
            if tag not in char_tags and tag not in ("general", "sensitive", "questionable", "explicit"):
                totals[tag] = totals.get(tag, 0) + p
    r.top_tags = [t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:15]]
    return r


def danbooru_to_names(tag: str, dic: Dictionary) -> tuple[str, str | None]:
    """raiden_shogun → 雷電將軍；onozuka_komachi → 小野塚小町；字典沒有就用原名（首字大寫）。"""
    m = re.match(r"^(?P<name>.+?)(?:_\((?P<series>[^)]+)\))?$", tag)
    name, series = m.group("name"), m.group("series")
    name = re.sub(r"_kai(?:_ni)?$", "", name)  # 艦これ改造型號：maya_kai_ni → maya
    work = None
    if series:
        works = dic.match_works(series.replace("_", " "))
        s_norm = normalize(series.replace("_", " "))
        not_work = re.search(r"costume|outfit|swimsuit|uniform|dress|bunny|cosplay|alternate|\d+(st|nd|rd|th)", s_norm) \
            or any(s_norm in e.aliases for e in dic.characters)
        work = works[0] if works else (None if not_work else series.replace("_", " ").title())
    hits = dic.match_characters(name.replace("_", " "), [work] if work else [])
    if work:
        hits = [h for h in hits if not h[1] or h[1] == work] or hits
    if hits:
        return hits[0][0], hits[0][1] or work
    pretty = " ".join(w.capitalize() for w in name.split("_"))
    return pretty, work


def candidates_for(ai: AIResult, dic: Dictionary, style_authors: list) -> dict:
    """人工確認時給你挑的候選與信心：類別、動畫類型各類的機率，角色前 5 名，作品，畫風相近的作者。"""
    chars, works = [], {}
    for c in ai.character_candidates or ai.characters:
        name, work = danbooru_to_names(c["tag"], dic)
        if all(name != x[0] for x in chars):
            chars.append([name, work, c["prob"]])
        if work:
            works[work] = max(works.get(work, 0), c["prob"])
    cat = dict(ai.category_probs or {})
    if not cat and ai.category:
        cat = {ai.category: 1.0 if ai.category_sure else 0.5}
    return {"category": {k: round(v, 3) for k, v in sorted(cat.items(), key=lambda kv: -kv[1])},
            "anim_type": {k: round(v, 3) for k, v in sorted((ai.style or {}).items(), key=lambda kv: -kv[1])},
            "characters": chars[:5],
            "works": [[w, round(p, 3)] for w, p in sorted(works.items(), key=lambda kv: -kv[1])][:5],
            "authors": [list(x) for x in style_authors][:3]}


def series_memory(cfg: Config) -> dict[str, list[dict]]:
    """收藏庫裡已歸檔（自動確定或人工確認）的影片，依作者記下標題與角色、作品；
    同一位作者、標題相近（同一系列或差分）的新影片就直接沿用角色。"""
    from collections import defaultdict
    mem: dict[str, list[dict]] = defaultdict(list)
    for v in _videos(cfg.layout.library, cfg):
        f = parse_final_name(v.name)
        if f and f.get("author") and f.get("characters"):
            mem[f["author"]].append({"title": f["title"], "characters": f["characters"],
                                     "works": f.get("works") or [], "file": v.name})
    return mem


def recall_series(info: dict, mem: dict[str, list[dict]], min_similarity: float = 0.8) -> dict | None:
    from vsort.hints import similarity
    best = None
    for item in mem.get(info.get("author") or "", []):
        s = similarity(info.get("title") or "", item["title"])
        if s >= min_similarity and (best is None or s > best[0]):
            best = (s, item)
    return best[1] if best else None


def author_styles(cfg: Config) -> dict[str, list[str]]:
    """從收藏庫統計每位作者的動畫類型；同一作者 ≥3 部且 ≥80% 一致，就當作他的慣用類型。
    你在人工確認改過的結果也會被算進來，所以越用越準。"""
    from collections import Counter, defaultdict
    stats: dict[str, Counter] = defaultdict(Counter)
    for v in _videos(cfg.layout.library, cfg):
        f = parse_final_name(v.name)
        if f and f.get("author") and f.get("anim_type"):
            stats[f["author"]][",".join(f["anim_type"])] += 1
    out = {}
    for author, c in stats.items():
        label, n = c.most_common(1)[0]
        total = sum(c.values())
        if total >= 3 and n / total >= 0.8:
            out[author] = label.split(",")
    return out


def character_authors(cfg: Config) -> dict[str, dict[str, int]]:
    """F25：收藏庫裡每個角色出現在哪些作者的作品，用來替不知道作者的影片提出建議。"""
    from collections import Counter, defaultdict
    stats: dict[str, Counter] = defaultdict(Counter)
    for v in _videos(cfg.layout.library, cfg):
        f = parse_final_name(v.name)
        if f and f.get("author"):
            for c in f.get("characters") or []:
                stats[c][f["author"]] += 1
    return {c: dict(cnt.most_common(5)) for c, cnt in stats.items()}


def suggest_authors(characters: list[str], char_authors: dict[str, dict[str, int]]) -> list[tuple[str, int]]:
    from collections import Counter
    total: Counter = Counter()
    for c in characters:
        total.update(char_authors.get(c, {}))
    return total.most_common(3)


def style_labels(ai: AIResult, opts: dict) -> tuple[list[str], bool]:
    """把動畫類型機率轉成標籤。分類器每個類型各自判斷（可複選）；零樣本只取最高的一個。"""
    from vsort.naming import anim_list
    if ai.style_kind == "classifier":
        base, margin = opts["style_label_threshold"], opts["style_label_margin"]
        per = opts.get("style_label_thresholds") or {}
        thr = {k: per.get(k, base) for k in ai.style}
        labels = [k for k, v in ai.style.items() if v >= thr[k]]
        if not labels:
            labels = [max(ai.style, key=ai.style.get)]
        # MMD 寧可多標：MMD 被標上時不算不確定；其他類型要離門檻夠遠才算確定。
        # 機率很低（≤ floor）就是明顯沒有：以前 MMD 門檻 0.3 配邊界 0.3，MMD 只要 0～30% 就一律算不確定，
        # 幾乎所有非 MMD 的影片都被送人工確認（30 部重測裡 11 部 → 5 部）。
        floor = opts.get("style_label_floor", 0.0)
        sure = all((v >= thr[k] and (k == "MMD" or v - thr[k] >= margin)) or (v < thr[k] and (v <= floor or thr[k] - v >= margin))
                   for k, v in ai.style.items())
        return anim_list(labels), sure
    label, prob = max(ai.style.items(), key=lambda kv: kv[1])
    return [label], prob >= opts["style_sure"]


def decide(sidecar: dict, ai: AIResult, dic: Dictionary, opts: dict,
           styles: dict[str, list[str]] | None = None,
           char_authors: dict[str, dict[str, int]] | None = None,
           priors=None) -> tuple[dict, bool, list[str]]:
    """合併各來源（F23：網路查詢 > AI > 檔名），決定是否「已確定」。回傳（info, sure, 理由）。
    priors：AuthorPriors（作者先驗）；給了就把類別機率乘上同作者過去的類別比例，動畫類型沿用該作者慣用類型。"""
    info = dict(sidecar.get("info") or {})
    reasons: list[str] = []
    hint = (sidecar.get("parsed") or {}).get("category_hint")
    if priors is not None and styles is None:
        styles = {}
        prior_style = priors.anim_type(info.get("author"))
        if prior_style:
            styles[info.get("author") or ""] = prior_style
    if priors is not None and ai.category_probs and info.get("author") and (opts.get("author_prior_weight") or 0) > 0:
        priors.weight = float(opts.get("author_prior_weight", 1.0))
        probs, used = priors.apply_category(ai.category_probs, info.get("author"))
        if used:
            ai.category_prior = used
            ai.category_probs = {k: round(v, 3) for k, v in probs.items()}
            ai.category = max(probs, key=probs.get)
            ai.category_sure = probs[ai.category] >= opts.get("category_sure_prob", 0.8)

    ai_chars, ai_works, unsure_chars = [], [], []
    for c in ai.characters:
        if c["tag"].startswith("male_"):  # 男性角色不列入
            continue
        name, work = danbooru_to_names(c["tag"], dic)
        if dic.is_male(name):
            continue
        ai_chars.append(name)
        if work:
            ai_works.append(work)
        if c["prob"] < opts["character_sure"]:
            unsure_chars.append(name)
    # 已經是英文名（例如重跑時從資訊檔來的）先換回字典標準名，避免中英重複
    info["characters"] = [dic.canonical_character(c) or c for c in (info.get("characters") or [])]
    had_chars = bool(info.get("characters"))
    # 檔名或網站已知的角色、作品排前面；AI 補的依信心排序，有把握的在前
    sure_ai = [c for c in ai_chars if c not in unsure_chars]
    info["characters"] = list(dict.fromkeys((info.get("characters") or []) + sure_ai + unsure_chars))
    info["works"] = list(dict.fromkeys((info.get("works") or []) + ai_works))
    if not had_chars and unsure_chars and not sure_ai:
        reasons.append(f"角色由 AI 推測，信心不高：{'、'.join(unsure_chars)}")
    # 至少一個主要角色有把握就可以歸檔；其他沒把握的記在資訊檔，日後篩選用
    info["characters_unconfirmed"] = [dic.character_names(c)["en"] for c in unsure_chars if had_chars or sure_ai]
    info["characters_incomplete"] = bool(info["characters_unconfirmed"])
    info["characters_i18n"] = [dic.character_names(c) for c in info["characters"] if not dic.is_male(c)]
    info["characters"] = dic.display_characters(info["characters"])  # 檔名用英文名

    prior = (styles or {}).get(info.get("author") or "")
    if not info.get("anim_type") and prior:
        info["anim_type"] = prior  # 這位作者收藏庫裡的作品幾乎都是這個類型
    if not info.get("anim_type") and ai.style:
        labels, sure_style = style_labels(ai, opts)
        info["anim_type"] = labels
        if not sure_style:
            others = "、".join(f"{k} {v:.0%}" for k, v in sorted(ai.style.items(), key=lambda kv: -kv[1]))
            reasons.append(f"動畫類型不確定（{others}）")

    info["category"] = ai.category
    if not ai.category_sure:
        if ai.category_probs:
            probs = "、".join(f"{k} {v:.0%}" for k, v in sorted(ai.category_probs.items(), key=lambda kv: -kv[1]))
            prior = f"，已含作者先驗 {ai.category_prior['n']} 部" if ai.category_prior else ""
            reasons.append(f"類別信心不足（{probs}；Qwen3-VL 判斷 {ai.category_vlm or '?'}，"
                           f"場景畫面 {ai.scene_ratio:.0%}{prior}）")
        elif ai.category_vlm and ai.category_tags and ai.category_vlm != ai.category_tags:
            reasons.append(f"類別：Qwen3-VL 判斷 {ai.category_vlm}，畫面標籤判斷 {ai.category_tags}"
                           f"（場景畫面 {ai.scene_ratio:.0%}）")
        else:
            reasons.append(f"類別在門檻附近（場景畫面 {ai.scene_ratio:.0%}）")
    if hint and ai.category and hint != ai.category:
        reasons.append(f"檔名看起來是 {hint}，AI 判斷是 {ai.category}")
    if ai.frames == 0:
        reasons.append("讀不到影片畫面")
    # 不知道作者不算「可能誤判」：其他欄位有把握就直接歸到收藏庫的「作者未知」資料夾，之後另外處理
    # （你 10/10 決定：不要依角色去猜作者）
    # 作品不列入自動歸檔的判斷（你 10/10 決定）：沒有就填信心最高的候選
    if not info.get("works"):
        for c in sorted(ai.character_candidates or ai.characters, key=lambda c: -c["prob"]):
            work = danbooru_to_names(c["tag"], dic)[1]
            if work:
                info["works"] = [work]
                break
    if not info.get("characters"):
        reasons.append("不知道角色")
    return info, not reasons, reasons


# ---------- AI 執行與快取 ----------

def file_key(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{st.st_size}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()[:20]


class Analyzer:
    """包裝模型與快取；tagger 延遲載入，沒有需要分析的影片就不載模型。"""

    def __init__(self, cfg: Config, opts: dict, tagger=None, extractor=None, style=None):
        self.opts = opts
        self.cache = cfg.layout.workspace / "_cache" / "ai"
        self._tagger = tagger
        self._style = style
        self._extract = extractor

    @property
    def style(self):
        """動畫類型模型：trained＝自訓小分類器（可複選）；含 siglip＝SigLIP 零樣本；其餘＝CLIP 零樣本。"""
        if self._style is None:
            name = self.opts["style_model"]
            if name == "trained":
                from vsort.style_train import TrainedStyleClassifier
                self._style = TrainedStyleClassifier(self.cache.parent)
            elif "siglip" in name.lower():
                from vsort.style_train import SiglipEmbedder
                emb = SiglipEmbedder(self.cache.parent, name)
                self._style = _ZeroShot(emb.zero_shot)
            else:
                from vsort.tagger import StyleClassifier
                clf = StyleClassifier(name)
                self._style = _ZeroShot(lambda imgs: _mean(clf.predict(imgs)))
        return self._style

    @property
    def describer(self):
        """額外的描述模型（只用在可能誤判的影片）。"""
        name = (self.opts.get("describe_model") or "none").lower()
        if name in ("none", "", "false"):
            return None
        return self._get_describer(name)

    def _get_describer(self, name: str):
        self._describers = getattr(self, "_describers", {})
        if getattr(self, "_describer", None) is not None and name == self.opts.get("describe_model"):
            return self._describer  # 測試注入用
        if name not in self._describers:
            from vsort.describe import DESCRIBERS
            self._describers[name] = DESCRIBERS[name]()
        return self._describers[name]

    def _style_model_id(self) -> str:
        """自訓分類器重新訓練後，舊的快取結果要作廢，所以把模型檔的修改時間算進去。"""
        name = self.opts["style_model"]
        if name == "trained":
            from vsort.style_train import CLASSIFIER_PATH
            if CLASSIFIER_PATH.exists():
                return f"trained@{int(CLASSIFIER_PATH.stat().st_mtime)}"
        return name

    def _category_model_id(self) -> str:
        name = self.opts.get("category_model", "tags")
        if name in ("qwen3vl", "combined"):
            from vsort.describe import PROMPT_VERSION
            ident = f"{name}@v{PROMPT_VERSION}"
            if name == "combined":
                from vsort.category_train import CATEGORY_CLASSIFIER, FALLBACK_CLASSIFIER
                if CATEGORY_CLASSIFIER.exists():
                    ident += f"@{int(CATEGORY_CLASSIFIER.stat().st_mtime)}"
                if FALLBACK_CLASSIFIER.exists() and self.opts.get("category_second_stage", True):
                    ident += f"+m{int(FALLBACK_CLASSIFIER.stat().st_mtime)}"
            return ident
        return name

    def _fallback_classifier(self):
        if not self.opts.get("category_second_stage", True):
            return None
        from vsort.category_train import FALLBACK_CLASSIFIER, CategoryClassifier
        if not FALLBACK_CLASSIFIER.exists():
            return None
        if getattr(self, "_fbclf", None) is None:
            self._fbclf = CategoryClassifier(FALLBACK_CLASSIFIER)
        return self._fbclf

    def _category_classifier(self):
        if self.opts.get("category_model") != "combined":
            return None
        from vsort.category_train import CATEGORY_CLASSIFIER, CategoryClassifier
        if not CATEGORY_CLASSIFIER.exists():
            return None
        if getattr(self, "_catclf", None) is None:
            self._catclf = CategoryClassifier()
        return self._catclf

    def _frames(self, video: Path, key: str):
        if self._extract:
            return self._extract(video, self.opts["frames"])
        from vsort.frames import extract_cached
        return extract_cached(video, self.opts["frames"], self.cache.parent / "frames" / key)

    def qwen_signals(self, video: Path, fallback: bool = False) -> tuple[dict, dict, dict]:
        """Qwen3-VL 的判斷，各自快取：逐張判斷 v1（統計）、影片模式、（選用）備援版逐張判斷 v3（統計）。"""
        key = file_key(video)
        jobs = [("frames1", self.cache / f"{key}_qwenframes_v1.json"),
                ("video", self.cache / f"{key}_qwenvideo_v1.json")]
        if fallback:
            jobs.append(("frames3", self.cache / f"{key}_qwenframes_v3.json"))
        out = {}
        for name, path in jobs:
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
                continue
            imgs = [im for _, im in self._frames(video, key)]
            q = self._get_describer("qwen3vl")
            if not imgs:
                r = {}
            elif name == "video":
                r = q.classify_video(imgs)
            else:
                r = q.classify_frames(imgs, version=1 if name == "frames1" else 3)
            self.cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            out[name] = r
        return ((out["frames1"].get("stats") or {}), out["video"],
                (out.get("frames3") or {}).get("stats") or {})

    def audio_features(self, video: Path, key: str | None = None) -> list[float]:
        """聲音特徵（librosa，不用模型），快取在 _cache/audio。"""
        from vsort.audio import cached_features
        return cached_features(video, self.cache.parent / "audio", key or file_key(video))

    def describe(self, video: Path, name: str | None = None) -> dict:
        name = name or self.opts["describe_model"]
        key = file_key(video)
        from vsort.describe import PROMPT_VERSION
        cpath = self.cache / f"{key}_describe_{name}_v{PROMPT_VERSION}.json"
        if cpath.exists():
            return json.loads(cpath.read_text(encoding="utf-8"))
        imgs = [im for _, im in self._frames(video, key)]
        d = self._get_describer(name).describe(imgs) if imgs else {}
        self.cache.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return d

    @property
    def tagger(self):
        if self._tagger is None:
            from vsort.tagger import make_tagger
            self._tagger = make_tagger(self.opts["model"])
        return self._tagger

    def run(self, video: Path, progress: Callable[[str], None] | None = None) -> AIResult:
        key = file_key(video)
        cpath = self.cache / f"{key}.json"
        if cpath.exists():
            try:
                cached = AIResult(**json.loads(cpath.read_text(encoding="utf-8")))
                if cached.model == self.opts["model"] and cached.style_model == self._style_model_id() \
                        and cached.category_model == self._category_model_id()                         and (cached.style or not cached.frames):
                    return cached
            except (OSError, ValueError, TypeError):
                pass
        if progress:
            progress(video.name)
        import time
        t0 = time.perf_counter()
        frames = self._frames(video, key)
        images = [img for _, img in frames]
        t1 = time.perf_counter()
        per_frame = self.tagger.predict(images) if frames else []
        res = summarize(per_frame, self.tagger.character_set(), self.opts)
        t2 = time.perf_counter()
        res.timings = {"frames": round(t1 - t0, 2), "tagger": round(t2 - t1, 2)}
        if images:
            style = self.style
            if hasattr(style, "predict_video"):
                res.style = {k: round(v, 3) for k, v in style.predict_video(key, images).items()}
                res.style_kind = "classifier"
            else:
                res.style = {k: round(v, 3) for k, v in style(images).items()}
        t3 = time.perf_counter()
        res.timings["style"] = round(t3 - t2, 2)
        res.category_tags = res.category
        if images and self.opts.get("category_model") in ("qwen3vl", "combined"):
            clf = self._category_classifier()
            if clf is not None and clf.feature_set != "tags+qwen_json":
                # 目前做法：Qwen 逐張判斷＋影片模式（比整部描述快也準），再和畫面標籤一起交給分類器
                from vsort.category_train import build_features
                fallback = "frames_v3" in clf.feature_set
                fs, vd, f3 = self.qwen_signals(video, fallback=fallback)
                res.category_vlm = vd.get("category")
                ta = time.perf_counter()
                au = self.audio_features(video, key) if "audio" in clf.feature_set else None
                res.timings["audio"] = round(time.perf_counter() - ta, 2)
                feats = build_features(clf.feature_set, res.scene_ratio, res.dance_ratio, res.explicit_ratio, fs, vd, f3, au)
            else:
                d = self.describe(video, "qwen3vl")
                res.description = d.get("text", "")
                res.category_vlm = (d.get("json") or {}).get("category")
                if clf is not None:
                    from vsort.category_train import features
                    feats = features(res.scene_ratio, res.dance_ratio, res.explicit_ratio, d.get("json"))
            if res.category_vlm not in ("Dance", "Mix", "Scene"):
                res.category_vlm = None
            vlm = res.category_vlm
            if clf is not None:
                probs = clf.predict(feats)
                sure_prob = self.opts.get("category_sure_prob", 0.8)
                mclf = self._fallback_classifier()
                if max(probs.values()) < sure_prob and mclf is not None and "frames_v3" not in clf.feature_set:
                    # 第二段：信心不足才加跑備援版逐張判斷
                    from vsort.category_train import build_features
                    fs, vd, f3 = self.qwen_signals(video, fallback=True)
                    au2 = self.audio_features(video, key) if "audio" in mclf.feature_set else None
                    probs = mclf.predict(build_features(mclf.feature_set, res.scene_ratio, res.dance_ratio,
                                                        res.explicit_ratio, fs, vd, f3, au2))
                    res.timings["fallback"] = True
                res.category = max(probs, key=probs.get)
                res.category_probs = {k: round(v, 3) for k, v in probs.items()}
                res.category_sure = probs[res.category] >= sure_prob
            elif res.category_vlm:
                # 類別以 Qwen3-VL 為準；和畫面標籤一致才算確定
                res.category_sure = res.category_vlm == res.category_tags
                res.category = res.category_vlm
        res.timings["category"] = round(time.perf_counter() - t3, 2)
        res.model = self.opts["model"]
        res.style_model = self._style_model_id()
        res.category_model = self._category_model_id()
        self.cache.mkdir(parents=True, exist_ok=True)
        if frames:
            from vsort.frames import contact_sheet
            prev = self.cache / f"{key}.jpg"
            contact_sheet(frames).save(prev, quality=80)
            res.preview = str(prev)
            frames[len(frames) // 2][1].save(self.cache / f"{key}_mid.jpg", quality=90)
        cpath.write_text(json.dumps(res.to_dict(), ensure_ascii=False), encoding="utf-8")
        _release_gpu_cache()  # 幾個模型同時在顯示卡上，釋放暫存避免記憶體滿了變慢
        return res


def _release_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class _ZeroShot:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, images):
        return self.fn(images)


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def options(cfg: Config) -> dict:
    return DEFAULTS | (cfg.data.get("ai") or {})


def _library_targets(cfg: Config) -> list[Path]:
    """F22：收藏庫裡類別或標籤不全的檔案也重新分析。"""
    out = []
    for v in _videos(cfg.layout.library, cfg):
        f = parse_final_name(v.name)
        if f is None or not f.get("category") or not f.get("author"):
            out.append(v)
    return out


def _reverse_search(cfg: Config, video: Path, searcher) -> dict | None:
    """以圖搜圖（預設關閉）：只對所有查詢都失敗、作者仍未知的影片，取 2 張關鍵幀查 Yandex。
    結果只當線索寫進資訊檔（看得出來源網站的網址、轉貼標題），不自動填作者。"""
    key = file_key(video)
    fdir = cfg.layout.workspace / "_cache" / "frames" / key
    imgs = sorted(fdir.glob("[0-9][0-9].jpg"))
    if not imgs:
        return None
    picks = list(dict.fromkeys([imgs[len(imgs) // 3], imgs[(2 * len(imgs)) // 3]]))
    from vsort.imgsearch import sources_in
    results = []
    for i, img in enumerate(picks):
        try:
            r = searcher.search(img, f"{key}_{i}")
        except Exception:  # 只是線索，失敗不影響主流程
            r = None
        results += r or []
    if not results:
        return None
    return {"engine": "yandex", "sources": [list(x) for x in sources_in([x.get("url", "") for x in results])],
            "titles": list(dict.fromkeys(x.get("title", "") for x in results if x.get("title")))[:10]}


def make_searcher(cfg: Config):
    """設定檔 reverse_search.enabled 為 true 才啟用；只用 Yandex（SauceNAO 實測對 MMD／3D 影片找不到來源，不排進路徑）。"""
    rc = cfg.data.get("reverse_search") or {}
    if not rc.get("enabled"):
        return None
    try:
        from vsort.imgsearch import YandexSearch
    except ImportError:
        return None
    return YandexSearch(cfg.layout.workspace / "_cache")


def plan_analyze(cfg: Config, analyzer: Analyzer, include_library: bool = False,
                 progress: Callable[[str], None] | None = None, searcher=None) -> tuple[Plan, dict]:
    lay = cfg.layout
    naming = cfg.naming
    dic = Dictionary(cfg.dictionary_dir)
    opts = analyzer.opts
    plan = Plan(kind="analyze")
    counts = {"library": 0, "review": 0}
    taken: set[str] = set()
    from vsort.priors import AuthorPriors
    priors = AuthorPriors(cfg)  # 作者先驗：標註資料＋收藏庫（含人工確認過的）
    series = series_memory(cfg)
    style_index = None
    targets = _videos(lay.refine, cfg) + _videos(lay.identify, cfg)
    if include_library:
        targets += _library_targets(cfg)
    for video in targets:
        prev = sc.read(video)
        data = dict(prev or {})
        if not data.get("info"):
            f = parse_final_name(video.name) or {"title": video.stem}
            data["info"] = f
        ai = analyzer.run(video, progress)
        prior = None
        if not (data["info"].get("characters")):
            prior = recall_series(data["info"], series)
            if prior:  # 同作者同系列已確認過的角色，直接沿用
                data = {**data, "info": {**data["info"], "characters": list(prior["characters"]),
                                         "works": list(dict.fromkeys((data["info"].get("works") or []) + prior["works"]))}}
        info, sure, reasons = decide(data, ai, dic, opts, priors=priors)
        description = None
        if not sure and analyzer.describer is not None and not (
                opts.get("category_model") in ("qwen3vl", "combined") and opts.get("describe_model") == "qwen3vl"):
            description = analyzer.describe(video)
            vlm_cat = (description.get("json") or {}).get("category")
            border = [r for r in reasons if r.startswith("類別在門檻附近")]
            if border and vlm_cat == ai.category:  # 描述模型也這樣判斷 → 類別算確定
                reasons = [r for r in reasons if r not in border]
                sure = not reasons
            elif vlm_cat and vlm_cat != ai.category:
                reasons.append(f"描述模型判斷是 {vlm_cat}")
        new = dict(data)
        new["ai"] = ai.to_dict()
        if description:
            new["description"] = description
        new["info"] = info
        new["certainty"] = "已確定" if sure else "可能誤判"
        new["reasons"] = reasons
        new["suggest_authors"] = []
        if not info.get("author"):  # 只當參考：畫風最像的作者，不影響歸檔
            if style_index is None:
                from vsort.style_author import StyleAuthorIndex
                style_index = StyleAuthorIndex(cfg)
            new["style_authors"] = style_index.suggest(file_key(video))
        new["candidates"] = candidates_for(ai, dic, new.get("style_authors") or [])
        if prior:
            new["series_from"] = prior["file"]
        lookup_ok = (data.get("lookup") or {}).get("status") == "ok"
        if searcher is not None and not info.get("author") and not lookup_ok:
            found = _reverse_search(cfg, video, searcher)
            if found:  # 以圖搜圖的結果只記在資訊檔，給之後處理未知作者時參考，不影響歸檔
                new["reverse_search"] = found
        new["stage"] = "library" if sure else "review"
        if ai.preview:
            new["preview"] = ai.preview
        new.setdefault("history", []).append(
            {"t": datetime.now().isoformat(timespec="seconds"), "step": "analyze", "stage": new["stage"]})
        name = build_final_name(info, video.suffix, naming["illegal_char_style"], naming["max_name_length"])
        if sure:
            dest = lay.library / sanitize_dirname(info.get("author") or UNKNOWN_AUTHOR_DIR) / name
        else:
            dest = lay.review / name
        dst = unique_path(dest, taken) if Path(dest) != video else video
        if dst == video:
            continue
        counts[new["stage"]] += 1
        plan.ops.append(Op(action="move", src=str(video), dst=str(dst), certainty=new["certainty"],
                           note="；".join(reasons) if reasons else f"類別 {ai.category}",
                           sidecar_mode="write", sidecar=new, sidecar_prev=prev))
    return plan, counts
