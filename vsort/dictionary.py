"""關鍵字字典（F06）與作者別名對照（F31）。

字典是 dict 資料夾底下的 CSV，可以直接用 Excel 編輯（存成「CSV UTF-8」）。
- works.csv：作品標準名, 別名
- characters.csv：角色標準名, 作品, 別名
- authors.csv：作者標準名, 網站, 帳號ID, 別名（同一位作者在不同網站可以有多列）
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_SEP = re.compile(r"[_\-.・·~～]+")


def normalize(text: str) -> str:
    """比對用：全形轉半形、轉小寫、底線與連字號視為空白。"""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _SEP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _script(ch: str) -> str:
    o = ord(ch)
    if ch.isascii():
        return "ascii" if ch.isalnum() else ""
    if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kata"
    if 0x3040 <= o <= 0x309F:
        return "hira"
    if 0xAC00 <= o <= 0xD7AF:
        return "hangul"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "han"
    return ""


def _find(alias: str, text: str) -> int:
    """在 text 中找 alias，要求前後不是同一種文字，避免「リン」命中「リンパ」。回傳位置或 -1。"""
    start = 0
    first, last = _script(alias[0]), _script(alias[-1])
    while True:
        i = text.find(alias, start)
        if i < 0:
            return -1
        before = text[i - 1] if i > 0 else ""
        after = text[i + len(alias)] if i + len(alias) < len(text) else ""
        ok = True
        # 漢字名稱（兩字以上）允許接在其他漢字旁，例如「夏祭申鶴」
        if first and first != "han" and before and _script(before) == first:
            ok = False
        if last and last != "han" and after and _script(after) == last:
            ok = False
        if ok:
            return i
        start = i + 1


@dataclass
class Entry:
    name: str
    work: str = ""
    aliases: list[str] = field(default_factory=list)
    en: str = ""        # 英文名（檔名用）
    ja: str = ""        # 日文名
    male: bool = False  # 男性角色不列入角色欄位


def _read_csv(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return [r for r in rows[1:] if r and r[0].strip() and not r[0].startswith("#")]


def _aliases(name: str, raw: str) -> list[str]:
    out = {normalize(name)}
    for a in raw.split("|"):
        a = normalize(a)
        if a:
            out.add(a)
    # 一個字的別名太容易誤判，忽略
    return sorted((a for a in out if len(a) >= 2), key=len, reverse=True)


@dataclass
class Author:
    name: str
    accounts: set[tuple[str, str]] = field(default_factory=set)  # (網站, 帳號ID)
    aliases: set[str] = field(default_factory=set)


class Dictionary:
    def __init__(self, folder: Path):
        self.folder = folder
        self.works: list[Entry] = []
        self.characters: list[Entry] = []
        self.authors: dict[str, Author] = {}
        self.load()

    def load(self) -> None:
        self.works = [Entry(r[0].strip(), "", _aliases(r[0], r[1] if len(r) > 1 else ""))
                      for r in _read_csv(self.folder / "works.csv")]
        self.characters = []
        for r in _read_csv(self.folder / "characters.csv"):
            r = r + [""] * (6 - len(r))
            en = r[3].strip()
            self.characters.append(Entry(r[0].strip(), r[1].strip(), _aliases(r[0], r[2] + ("|" + en if en else "")),
                                         en=en, ja=r[4].strip(), male=r[5].strip() in ("1", "是", "男", "y", "Y")))
        self._by_name = {e.name: e for e in self.characters}
        self.authors = {}
        for r in _read_csv(self.folder / "authors.csv"):
            r = r + [""] * (4 - len(r))
            name = r[0].strip()
            a = self.authors.setdefault(name, Author(name))
            if r[1].strip() and r[2].strip():
                a.accounts.add((r[1].strip().lower(), r[2].strip()))
            a.aliases.add(normalize(name))
            for alias in r[3].split("|"):
                if alias.strip():
                    a.aliases.add(normalize(alias))

    # ---------- 比對 ----------

    def match_works(self, text: str) -> list[str]:
        t = normalize(text)
        found = []
        for e in self.works:
            if any(_find(a, t) >= 0 for a in e.aliases) and e.name not in found:
                found.append(e.name)
        return found

    def match_characters(self, text: str, works: list[str]) -> list[tuple[str, str]]:
        """回傳 (角色, 作品)。同名角色出現在多部作品時，以已知作品決定；無法決定就不填作品。"""
        t = normalize(text)
        hits: list[tuple[int, Entry]] = []
        for e in self.characters:
            pos = min((p for p in (_find(a, t) for a in e.aliases) if p >= 0), default=-1)
            if pos >= 0:
                hits.append((pos, e))
        hits.sort(key=lambda h: h[0])
        out: list[tuple[str, str]] = []
        for _, e in hits:
            if any(name == e.name for name, _ in out):
                continue
            work = e.work
            if not work:
                same = [x.work for _, x in hits if x.name == e.name and x.work]
                work = next((w for w in same if w in works), "")
            out.append((e.name, work))
        return out

    # ---------- 角色名稱：中文標準名 ↔ 英文名（檔名用）↔ 日文名 ----------

    def canonical_character(self, name: str) -> str | None:
        """任何語言的角色名 → 字典的中文標準名；不在字典裡回傳 None。"""
        if name in self._by_name:
            return name
        n = normalize(name)
        for e in self.characters:
            if n in e.aliases:
                return e.name
        return None

    def character_names(self, name: str) -> dict:
        zh = self.canonical_character(name)
        e = self._by_name.get(zh) if zh else None
        if e is None:
            return {"en": name, "zh": None, "ja": None}
        return {"en": e.en or e.name, "zh": e.name, "ja": e.ja or None}

    def is_male(self, name: str) -> bool:
        zh = self.canonical_character(name)
        return bool(zh and self._by_name[zh].male)

    def display_characters(self, names: list[str]) -> list[str]:
        """檔名用的角色名（英文），去掉重複與男性角色，保留順序。"""
        out = []
        for n in names or []:
            if self.is_male(n):
                continue
            en = self.character_names(n)["en"]
            if en not in out:
                out.append(en)
        return out

    def author_by_name(self, name: str) -> str | None:
        n = normalize(name)
        for a in self.authors.values():
            if n in a.aliases:
                return a.name
        return None

    def author_by_account(self, site: str, account_id: str) -> str | None:
        key = (site.lower(), str(account_id))
        for a in self.authors.values():
            if key in a.accounts:
                return a.name
        return None
