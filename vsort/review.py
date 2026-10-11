"""人工確認區（F24）：05_Review 的影片列成一份 CSV，用 Excel 修改欄位、在「確認」欄填 Y，
再匯入就會依修改後的欄位命名並歸檔到 04。之後的里程碑會做成網頁介面（F27）。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from vsort import sidecar as sc
from vsort.config import Config
from vsort.journal import Op, Plan
from vsort.naming import anim_list, build_final_name, parse_final_name, sanitize_dirname, unique_path
from vsort.process import _videos

HEADER = ["確認（Y＝歸檔）", "檔名", "標題", "作者", "作品", "角色", "動畫類型", "類別", "判斷理由", "預覽圖"]
CATEGORIES = {"dance": "Dance", "mix": "Mix", "scene": "Scene"}


def csv_path(cfg: Config) -> Path:
    return cfg.layout.review / "review.csv"


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.replace("，", ",").split(",") if x.strip() and x.strip() != "-"]


def _row_for(video: Path) -> list[str]:
    data = sc.read(video) or {}
    info = data.get("info") or parse_final_name(video.name) or {"title": video.stem}
    prev = sc.preview_for(video)
    return ["", video.name, info.get("title") or "", info.get("author") or "",
            ",".join(info.get("works") or []), ",".join(info.get("characters") or []),
            ",".join(anim_list(info.get("anim_type"))), info.get("category") or "",
            "；".join(data.get("reasons") or []), str(prev) if prev.exists() else ""]


def export(cfg: Config) -> tuple[Path, int]:
    """產生或更新 review.csv；已經填過的列保留你的修改。"""
    path = csv_path(cfg)
    old: dict[str, list[str]] = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in list(csv.reader(fh))[1:]:
                if len(r) >= 2:
                    old[r[1]] = r + [""] * (len(HEADER) - len(r))
    rows = [old.get(v.name) or _row_for(v) for v in _videos(cfg.layout.review, cfg)]
    cfg.layout.review.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    tmp.replace(path)
    return path, len(rows)


def normalize_info(raw: dict, video: Path) -> dict:
    """把 CSV 或網頁送來的欄位整理成標準格式。"""
    def lst(v):
        return _split(v) if isinstance(v, str) else [x for x in (v or []) if x and x != "-"]

    cat = (raw.get("category") or "").strip()
    return {"title": (raw.get("title") or "").strip() or video.stem,
            "author": (raw.get("author") or "").strip() or None,
            "works": lst(raw.get("works")), "characters": lst(raw.get("characters")),
            "anim_type": anim_list(raw.get("anim_type")),
            "category": CATEGORIES.get(cat.lower(), cat or None)}


def confirm_plan(cfg: Config, items: list[tuple[Path, dict]]) -> Plan:
    """依人工確認的欄位命名並歸檔到 04。角色名一律換成英文（字典有的話），中日文名記在資訊檔。"""
    from vsort.dictionary import Dictionary
    dic = Dictionary(cfg.dictionary_dir)
    plan = Plan(kind="review")
    naming = cfg.naming
    taken: set[str] = set()
    for video, raw in items:
        if not video.exists():
            plan.skipped.append((str(video), "檔案不在 05_Review 了"))
            continue
        info = normalize_info(raw, video)
        info["characters_i18n"] = [dic.character_names(c) for c in info["characters"]]
        info["characters"] = dic.display_characters(info["characters"])
        info["characters_unconfirmed"], info["characters_incomplete"] = [], False
        prev = sc.read(video)
        data = dict(prev or {})
        data.update(info=info, manual=info, certainty="人工確認", reasons=[], stage="library")
        data.setdefault("history", []).append(
            {"t": datetime.now().isoformat(timespec="seconds"), "step": "review", "stage": "library"})
        name = build_final_name(info, video.suffix, naming["illegal_char_style"], naming["max_name_length"])
        from vsort.analyze import UNKNOWN_AUTHOR_DIR
        dst = unique_path(cfg.layout.library / sanitize_dirname(info["author"] or UNKNOWN_AUTHOR_DIR) / name, taken)
        plan.ops.append(Op(action="move", src=str(video), dst=str(dst), certainty="人工確認",
                           sidecar_mode="write", sidecar=data, sidecar_prev=prev))
    return plan


def plan_import(cfg: Config) -> Plan:
    path = csv_path(cfg)
    if not path.exists():
        return Plan(kind="review")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    items = []
    for r in rows:
        r = r + [""] * (len(HEADER) - len(r))
        if r[0].strip().upper() != "Y":
            continue
        items.append((cfg.layout.review / r[1], {"title": r[2], "author": r[3], "works": r[4],
                                                  "characters": r[5], "anim_type": r[6], "category": r[7]}))
    return confirm_plan(cfg, items)
