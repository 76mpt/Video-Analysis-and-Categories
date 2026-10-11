"""作者別名對照（F31）：用各站帳號 ID 認人，名稱只是別名。

- 查詢到沒見過的帳號，而且名稱跟已知作者都不像 → 自動加進 dict\\authors.csv（新作者）。
- 帳號沒見過，但名稱跟已知作者相似、或檔案放在已知作者的資料夾裡 → 不自動合併，
  列進 dict\\authors_pending.csv 等你確認；在「確認」欄填 Y（接受）或 N（拒絕）後執行
  `vsort authors --accept`。
"""

from __future__ import annotations

import csv
import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from vsort import sidecar as sc
from vsort.config import Config
from vsort.dictionary import Dictionary, normalize

AUTHORS_HEADER = ["標準名", "網站", "帳號ID", "別名（用 | 分隔）"]
PENDING_HEADER = ["確認（Y 接受／N 拒絕）", "建議標準名", "網站", "帳號ID", "網站上的名稱",
                  "相似的已知作者", "原因", "檔案例"]


@dataclass
class Candidate:
    canonical: str
    site: str
    account_id: str
    name: str
    example: str


def similar(a: str, b: str) -> bool:
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y:
        return True
    if min(len(x), len(y)) >= 3 and (x in y or y in x):
        return True
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.8


def _read(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def collect_candidates(cfg: Config) -> list[Candidate]:
    lay = cfg.layout
    seen: set[tuple[str, str]] = set()
    out = []
    for folder in (lay.staging, lay.identify, lay.refine, lay.review, lay.library):
        if not folder.exists():
            continue
        for scp in folder.rglob("*" + sc.SUFFIX):
            try:
                data = json.loads(scp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            cand = (data.get("lookup") or {}).get("author_candidate")
            if not cand or not cand.get("account_id"):
                continue
            key = (cand["site"], str(cand["account_id"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(cand.get("canonical") or cand.get("name") or "", cand["site"],
                                 str(cand["account_id"]), cand.get("name") or "",
                                 scp.name[: -len(sc.SUFFIX)]))
    return out


def update_authors(cfg: Config) -> tuple[int, int]:
    """把新帳號加進別名表或待確認清單。回傳（自動新增數, 待確認數）。"""
    folder = cfg.dictionary_dir
    dic = Dictionary(folder)
    authors_path, pending_path = folder / "authors.csv", folder / "authors_pending.csv"
    authors = _read(authors_path)[1:]
    pending = _read(pending_path)[1:]
    pending_keys = {(r[2], r[3]) for r in pending if len(r) > 3}
    added = queued = 0
    for c in collect_candidates(cfg):
        if dic.author_by_account(c.site, c.account_id) or (c.site, c.account_id) in pending_keys:
            continue
        known_by_folder = dic.author_by_name(c.canonical)
        lookalikes = [a.name for a in dic.authors.values()
                      if any(similar(c.name or c.canonical, alias) for alias in a.aliases)
                      or similar(c.canonical, a.name)]
        if known_by_folder:
            reason = "檔案在已知作者的資料夾裡，帳號是新的"
        elif lookalikes:
            reason = "名稱和已知作者相似"
        else:
            alias = c.name if c.name and normalize(c.name) != normalize(c.canonical) else ""
            authors.append([c.canonical, c.site, c.account_id, alias])
            dic = _reload_with(dic, folder, authors)
            added += 1
            continue
        pending.append(["", known_by_folder or c.canonical, c.site, c.account_id, c.name,
                        "|".join(dict.fromkeys(([known_by_folder] if known_by_folder else []) + lookalikes)),
                        reason, c.example])
        pending_keys.add((c.site, c.account_id))
        queued += 1
    if added:
        _write(authors_path, AUTHORS_HEADER, authors)
    if queued:
        _write(pending_path, PENDING_HEADER, pending)
    return added, queued


def _reload_with(dic: Dictionary, folder: Path, rows: list[list[str]]) -> Dictionary:
    _write(folder / "authors.csv", AUTHORS_HEADER, rows)
    return Dictionary(folder)


def accept_pending(cfg: Config) -> tuple[int, int]:
    """套用待確認清單：Y 的寫進別名表並移出清單，N 的留在清單當紀錄。回傳（接受數, 拒絕數）。"""
    folder = cfg.dictionary_dir
    authors_path, pending_path = folder / "authors.csv", folder / "authors_pending.csv"
    authors = _read(authors_path)[1:]
    pending = _read(pending_path)[1:]
    keep, acc, rej = [], 0, 0
    for r in pending:
        r = r + [""] * (len(PENDING_HEADER) - len(r))
        mark = r[0].strip().upper()
        if mark == "Y":
            canonical, site, account, name = r[1].strip(), r[2], r[3], r[4]
            alias = name if name and normalize(name) != normalize(canonical) else ""
            authors.append([canonical, site, account, alias])
            acc += 1
        elif mark == "N":
            rej += 1
            keep.append(r)  # 留著當紀錄，之後不會再問同一個帳號
        else:
            keep.append(r)
    _write(authors_path, AUTHORS_HEADER, authors)
    _write(pending_path, PENDING_HEADER, keep)
    return acc, rej
