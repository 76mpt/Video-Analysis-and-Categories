"""檔名解析（F03 判斷檔名類型、F04 提取識別碼與清洗、F05 tag 提取）。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from vsort.dictionary import Dictionary, normalize

# 檔名類型（對應規劃的四種）
URL_ID = "url_id"          # 網域/video/子網址：檔名裡有來源網站與作品 ID
CODE_NAME = "code_name"    # 隨機碼＋作者檔名：例如 fanbox 下載的 88aa854e_標題
CODE_SUBID = "code_subid"  # 隨機碼_子網址
PLAIN = "plain"            # 僅有檔名
TYPE_LABEL = {URL_ID: "網址ID", CODE_NAME: "隨機碼+檔名", CODE_SUBID: "隨機碼+子網址", PLAIN: "僅有檔名"}

SITE_PATTERNS = [
    # Iwara 下載器格式：Iwara - 標題 [影片ID] [Source]
    ("iwara", re.compile(r"^iwara\s*-\s*(?P<title>.*?)\s*\[(?P<id>[A-Za-z0-9]{12,20})\]\s*(?:\[(?:source|\d{3,4}p?)\])?\s*$", re.I)),
    ("iwara", re.compile(r"(?:www\.)?iwara\.tv[\s_/\-]+videos?[\s_/\-]+(?P<id>[A-Za-z0-9]{8,20})", re.I)),
    ("fantia", re.compile(r"fantia(?:\.jp)?[\s_/\-]+posts?[\s_/\-]+(?P<id>\d{4,})", re.I)),
    ("patreon", re.compile(r"patreon(?:\.com)?[\s_/\-]+posts?[\s_/\-]+(?:[\w\-]*?-)?(?P<id>\d{6,})", re.I)),
    ("fanbox", re.compile(r"fanbox(?:\.cc)?[\s_/\-]+(?:@?[\w\-]+[\s_/\-]+)?posts?[\s_/\-]+(?P<id>\d{4,})", re.I)),
]
IWARA_PREFIX = re.compile(r"^iwara\s*-\s*(?P<title>.+)$", re.I)
HEX_CODE = re.compile(r"^(?P<code>[0-9a-f]{8})_(?P<rest>.+)$")
HEX_SUBID = re.compile(r"^(?P<code>[0-9a-zA-Z]{6,16})_(?P<id>\d{5,})$")
NUM_PREFIX = re.compile(r"^(?P<num>\d{6})(?=[\s_\-\[【(]|$)\s*[-_]?\s*(?P<rest>.*)$")
BRACKETS = re.compile(r"【([^】]{1,40})】|\[([^\]]{1,40})\]|［([^］]{1,40})］")

NOISE = re.compile(
    r"(?<![A-Za-z0-9])(?:source|(?:\d{3,4}p|[248]k)(?:\s?p?\d{2,3}fps)?|uhd|fhd|full ?hd|hd|"
    r"\d{2,3}fps|\d{3,4}x\d{3,4}|x264|x265|h\.?264|h\.?265|hevc|av1|nvenc)(?![A-Za-z0-9])", re.I)
# 7～8 位數字開頭：常見於 fantia／fanbox 下載工具，以貼文 ID 開頭
POST_PREFIX = re.compile(r"^(?P<num>\d{7,9})(?:[\s_\-]+(?P<rest>.*))?$")
# fanbox 下載工具的資料夾：貼文ID-日期-標題
FANBOX_FOLDER = re.compile(r"^(?P<id>\d{6,10})-\d{4}-\d{2}-\d{2}-(?P<title>.*)$")

DEFAULT_ANIM_RULES = {
    "MMD": [r"\bmmd\b", r"mikumikudance", r"skbmmd", r"sour式", r"\bmme\b"],
    "Koikatsu": [r"koikatsu", r"コイカツ", r"恋活", r"戀活", r"\bkk\b"],
    "VAM": [r"\bvam\b", r"virt ?a ?mate"],
    "3D": [r"\bhs2\b", r"honey ?select", r"blender", r"\bsfm\b", r"\b3d\b", r"unreal", r"\bue5\b"],
    "2D": [r"live2d", r"\b2d\b", r"手描き", r"手書き", r"\bhmv\b"],
}
# Category hints from filenames (the AI decides in the end); order = priority.
# Mix / Scene keyword rules are set per library under parser.category_hints in config.yaml.
DEFAULT_CATEGORY_HINTS = {
    "Dance": [r"dance only", r"\bnormal\b"],
}
DEFAULT_GENERIC_FOLDERS = [
    r"^作者未知$",  # 收藏庫裡還不知道作者的影片放這裡，再丟進來分類時不能當作者

    r"^未分類$", r"^未整理$", r"^未分类$", r"^用$", r"^iwara$", r"^hmv$", r"^downloads?$",
    r"^新增資料夾", r"^new folder", r"^temp$", r"^misc$", r"^others?$", r"^其他$",
    r"^\d+$", r"^\d{6}\b", r"^\d{4}-\d{2}", r"^tier \d", r"^video$", r"^videos$",
]


@dataclass
class Parsed:
    original: str
    name_type: str = PLAIN
    site: str | None = None
    source_id: str | None = None
    site_guess: str | None = None  # 沒有確定來源，但格式像某個網站
    code: str | None = None
    date: str | None = None
    title: str = ""
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    author_source: str | None = None  # folder / dictionary
    author_known: bool = False
    works: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    anim_type: list[str] = field(default_factory=list)  # 可複選：MMD、3D、2D
    category_hint: str | None = None
    folder_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def identified(self) -> bool:
        return bool(self.author or self.works or self.characters or self.source_id)


def _valid_yymmdd(s: str) -> bool:
    try:
        datetime.strptime(s, "%y%m%d")
        return True
    except ValueError:
        return False


class Parser:
    def __init__(self, dictionary: Dictionary, cfg: dict | None = None):
        cfg = cfg or {}
        self.dic = dictionary
        self.anim_rules = {k: [re.compile(p, re.I) for p in v]
                           for k, v in (cfg.get("anim_rules") or DEFAULT_ANIM_RULES).items()}
        self.category_hints = {k: [re.compile(p, re.I) for p in v]
                               for k, v in (cfg.get("category_hints") or DEFAULT_CATEGORY_HINTS).items()}
        self.generic = [re.compile(p, re.I)
                        for p in (cfg.get("generic_folders") or DEFAULT_GENERIC_FOLDERS)]

    def is_generic_folder(self, name: str) -> bool:
        return any(p.search(name.strip()) for p in self.generic)

    def _is_work_name(self, name: str) -> bool:
        """資料夾名稱本身就是作品名（例如 touhou），不當作者。"""
        n = normalize(name)
        return any(n in e.aliases for e in self.dic.works)

    def parse(self, filename: str, folders: list[str] | None = None) -> Parsed:
        """filename：檔名；folders：檔案在 Inbox 內的上層資料夾（由外到內）。"""
        folders = folders or []
        stem = Path(filename).stem
        p = Parsed(original=filename)
        title = stem

        # F03/F04：判斷類型並取出 ID
        for site, pat in SITE_PATTERNS:
            m = pat.search(stem)
            if m:
                p.name_type, p.site, p.source_id = URL_ID, site, m.group("id")
                title = m.group("title") if "title" in pat.groupindex else (stem[:m.start()] + " " + stem[m.end():])
                break
        else:
            if m := IWARA_PREFIX.match(stem):
                p.site, title = "iwara", m.group("title")
            if m := HEX_SUBID.match(title):
                p.name_type, p.code, p.source_id = CODE_SUBID, m.group("code"), m.group("id")
                title = ""
            elif m := HEX_CODE.match(title):
                p.name_type, p.code, title = CODE_NAME, m.group("code"), m.group("rest")
            elif (m := NUM_PREFIX.match(title)) and _valid_yymmdd(m.group("num")):
                p.date = datetime.strptime(m.group("num"), "%y%m%d").strftime("%Y-%m-%d")
                title = m.group("rest")
            elif m := POST_PREFIX.match(title):
                p.name_type, p.code = CODE_NAME, m.group("num")
                p.site_guess = "fantia"
                title = m.group("rest") or ""
        for f in folders:
            if (m := FANBOX_FOLDER.match(f)) and not p.source_id:
                p.site, p.source_id = "fanbox", m.group("id")
                p.site_guess = None
                p.name_type = URL_ID

        # F05：括號 tag
        p.tags = [next(g for g in m.groups() if g).strip() for m in BRACKETS.finditer(title)]

        # 作者：由外往內第一個不是通用名稱（未分類、iwara、日期…）的資料夾
        # （平常就以作者名建資料夾）。資料夾名稱若是作品名則當作品線索。
        rest_folders = list(folders)
        for i, f in enumerate(folders):
            if self.is_generic_folder(f):
                continue
            if self._is_work_name(f):
                break
            known = self.dic.author_by_name(f)
            p.author = known or f.strip()
            p.author_known = known is not None
            p.author_source = "folder"
            rest_folders = folders[:i] + folders[i + 1:]
            break
        p.folder_hints = [f for f in rest_folders if not self.is_generic_folder(f)]

        # 字典比對：標題＋tag＋其餘資料夾名稱
        haystack = " ".join([title, *p.folder_hints])
        p.works = self.dic.match_works(haystack)
        chars = self.dic.match_characters(haystack, p.works)
        p.characters = [c for c, _ in chars]
        for _, work in chars:
            if work and work not in p.works:
                p.works.append(work)

        norm = normalize(" ".join([stem, *p.folder_hints]))
        p.anim_type = [label for label, pats in self.anim_rules.items() if any(r.search(norm) for r in pats)]
        # 類別提示：檔名優先，檔名看不出來才看資料夾名稱
        for text in (normalize(stem), normalize(" ".join(p.folder_hints))):
            for label, pats in self.category_hints.items():
                if any(r.search(text) for r in pats):
                    p.category_hint = label
                    break
            if p.category_hint:
                break

        p.title = self.clean_title(title, p) or self.clean_title(stem, p) or stem
        return p

    def clean_title(self, title: str, p: Parsed) -> str:
        t = unicodedata.normalize("NFKC", title).replace("_", " ")

        # 移除已辨識為作品或動畫類型的括號 tag，其餘保留
        work_aliases = {a for e in self.dic.works for a in e.aliases}

        def drop(m: re.Match) -> str:
            inner = next(g for g in m.groups() if g)
            n = normalize(inner)
            if n in work_aliases:  # 【原神】
                return " "
            if len(n) <= 8 and any(r.search(n) for pats in self.anim_rules.values() for r in pats):
                return " "  # 【MMD】
            if NOISE.fullmatch(inner.strip()):
                return " "  # [4K]
            return m.group(0)

        t = BRACKETS.sub(drop, t)
        t = NOISE.sub(" ", t)
        t = re.sub(r"\(\s*\)|（\s*）|\[\s*\]", " ", t)
        t = re.sub(r"\s*\(\d{1,2}\)\s*$", "", t)  # 下載重複的 (1)
        t = re.sub(r"\s+", " ", t)
        return t.strip(" -_–—.、,")
