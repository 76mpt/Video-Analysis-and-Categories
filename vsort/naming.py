"""檔名安全處理（F12）：非法字元、長度、保留字、同名加序號。"""

from __future__ import annotations

import os
import re
from pathlib import Path

FULLWIDTH = {
    "<": "＜", ">": "＞", ":": "：", '"': "＂", "/": "／",
    "\\": "＼", "|": "｜", "?": "？", "*": "＊",
}
RESERVED = {"CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SPACES = re.compile(r"\s+")


def sanitize_stem(stem: str, style: str = "fullwidth", max_len: int = 180) -> str:
    """清理不含副檔名的檔名主體。"""
    s = _CONTROL.sub(" ", stem)
    if style == "underscore":
        s = "".join("_" if c in FULLWIDTH else c for c in s)
    else:
        s = "".join(FULLWIDTH.get(c, c) for c in s)
    s = _SPACES.sub(" ", s).strip()
    s = s.rstrip(". ")
    if len(s) > max_len:
        s = s[:max_len].rstrip(". ")
    if not s:
        s = "untitled"
    if s.split(".")[0].upper() in RESERVED:
        s = s + "_"
    return s


def sanitize_filename(name: str, style: str = "fullwidth", max_len: int = 180) -> str:
    """清理完整檔名，保留副檔名（轉小寫）。"""
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    if not re.fullmatch(r"\.[0-9a-z]{1,8}", ext):
        stem, ext = name, ""
    return sanitize_stem(stem, style, max_len) + ext


def sanitize_dirname(name: str, style: str = "fullwidth", max_len: int = 120) -> str:
    return sanitize_stem(name, style, max_len)


def _key(p: Path) -> str:
    return os.path.normcase(os.path.abspath(p))


def fit_path(path: Path, max_path: int = 250) -> Path:
    """整條路徑太長時截短檔名主體。"""
    excess = len(str(path)) - max_path
    if excess <= 0:
        return path
    stem, ext = path.stem, path.suffix
    keep = max(len(stem) - excess, 20)
    return path.with_name(stem[:keep].rstrip(". ") + ext)


def unique_path(path: Path, taken: set[str] | None = None) -> Path:
    """若目的地已存在（或已被本次計畫佔用），加上 (2)、(3)… 序號。

    taken 是本次計畫已分配出去的路徑鍵，會被更新。
    """
    taken = taken if taken is not None else set()
    candidate, n = path, 1
    while candidate.exists() or _key(candidate) in taken:
        n += 1
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
    taken.add(_key(candidate))
    return candidate


# ---------- 最終命名（F16）----------

MISSING = "-"
# 動畫類型（可複選）：Koikatsu＝戀活、VAM＝Virt-A-Mate、3D＝其他 3D（Blender、SFM…）、
# Audio＝聲音作品附贈、沒有動畫的影片（標註網頁上顯示「無」）
ANIM_ORDER = ["MMD", "Koikatsu", "VAM", "3D", "2D", "Audio"]
ANIM_DISPLAY = {"Audio": "無（聲音）"}


def anim_list(value) -> list[str]:
    """動畫類型可以複選（例如 Blender 做的 MMD：MMD,3D）。接受字串或清單，回傳固定順序的清單。"""
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).replace("，", ",").split(",")
    seen = [x.strip() for x in items if x and x.strip() and x.strip() != MISSING]
    aliases = {"戀活": "Koikatsu", "恋活": "Koikatsu", "kk": "Koikatsu", "無": "Audio", "无": "Audio",
               "none": "Audio", "virt-a-mate": "VAM"}
    seen = [aliases.get(x.lower(), x) for x in seen]
    known = [a for a in ANIM_ORDER if any(a.lower() == x.lower() for x in seen)]
    return known + [x for x in seen if x.lower() not in {a.lower() for a in ANIM_ORDER}]


def _field(value) -> str:
    if not value:
        return MISSING
    items = value if isinstance(value, (list, tuple)) else [value]
    cleaned = []
    for v in items:
        v = str(v).strip().replace("[", "［").replace("]", "］").replace(",", "，")
        if v:
            cleaned.append(v)
    return ",".join(cleaned) or MISSING


MAX_CHARACTERS_IN_NAME = 3  # 檔名最多列 3 個角色，完整名單在資訊檔


def tag_suffix(info: dict) -> str:
    """[作者][作品][角色][動畫類型][類別]，缺的欄位填 [-]，順序固定方便日後依位置切開。"""
    def val(k):
        if k == "anim_type":
            return anim_list(info.get(k))
        if k == "characters":
            return list(info.get(k) or [])[:MAX_CHARACTERS_IN_NAME]
        return info.get(k)
    return "".join(f"[{_field(val(k))}]" for k in ("author", "works", "characters", "anim_type", "category"))


def build_final_name(info: dict, ext: str, style: str = "fullwidth", max_len: int = 180) -> str:
    """標題 [作者][作品][角色][動畫類型][類別].ext；太長時只截短標題，標籤保留。"""
    suffix = sanitize_stem(tag_suffix(info), style, 10_000)
    budget = max(max_len - len(suffix) - 1, 10)
    title = sanitize_stem(str(info.get("title") or "untitled"), style, budget)
    return f"{title} {suffix}{ext.lower()}"


def parse_final_name(name: str) -> dict | None:
    """把最終檔名切回欄位（日後搜尋、篩選用）。"""
    m = re.match(r"^(?P<title>.*) \[(?P<author>[^\]]*)\]\[(?P<works>[^\]]*)\]\[(?P<characters>[^\]]*)\]"
                 r"\[(?P<anim_type>[^\]]*)\]\[(?P<category>[^\]]*)\](?: \(\d+\))?(?:\.\w+)?$", name)
    if not m:
        return None
    d = m.groupdict()
    out = {"title": d["title"]}
    for k in ("author", "category"):
        out[k] = None if d[k] == MISSING else d[k]
    out["anim_type"] = anim_list(d["anim_type"])
    for k in ("works", "characters"):
        out[k] = [] if d[k] == MISSING else d[k].split(",")
    return out
