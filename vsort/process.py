"""第一階段：檔名解析與分流（F08），把 01_Inbox（與 02 待重試）的影片送進 03 的兩個佇列。

所有檔案都要經過 AI 判定才會進 04（你 10/10 的決定），所以這一步只會把檔案送到：
- Refine_Queue：已經知道作者、作品、角色或來源 ID 其中之一，等 AI 補類別與缺的欄位
- Identify_Queue：檔名完全看不出是誰，等 AI 辨識
- 02_Parsed_Staging：網路查詢暫時失敗（之後的里程碑），下次再試
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from vsort import sidecar as sc
from vsort.config import Config
from vsort.dictionary import Dictionary
from vsort.journal import Op, Plan
from vsort.naming import anim_list, build_final_name, unique_path
from vsort.parser import Parsed, Parser

INFO_KEYS = ("title", "author", "works", "characters", "anim_type", "category")


@dataclass
class Enrichment:
    """網路查詢等步驟的結果。retry=True 代表暫時失敗，檔案留在 02 下次再試。"""
    info: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    retry: bool = False
    note: str = ""


Enricher = Callable[[Parsed, dict], Enrichment | None]


def make_parser(cfg: Config) -> Parser:
    return Parser(Dictionary(cfg.dictionary_dir), cfg.data.get("parser") or {})


def info_from_parsed(p: Parsed) -> dict:
    return {"title": p.title, "author": p.author, "works": list(p.works),
            "characters": list(p.characters), "anim_type": list(p.anim_type), "category": None}


def merge_info(base: dict, update: dict) -> dict:
    """後來的資訊（網路查詢、AI）覆蓋檔名解析；清單欄位合併。"""
    out = dict(base)
    for k, v in update.items():
        if k in ("works", "characters", "anim_type"):
            if k == "anim_type":
                v = anim_list(v)
            merged = list(dict.fromkeys(v or []))
            for x in (anim_list(out.get(k)) if k == "anim_type" else out.get(k) or []):
                if x not in merged:
                    merged.append(x)
            out[k] = anim_list(merged) if k == "anim_type" else merged
        elif v:
            out[k] = v
    return out


def _videos(folder: Path, cfg: Config) -> list[Path]:
    """folder 底下所有影片。工作區（sorting）若在這個資料夾底下（例如收藏庫就是根目錄）會整個略過。"""
    if not folder.exists():
        return []
    exts, temps = cfg.video_extensions, cfg.temp_extensions
    ws = cfg.layout.workspace
    skip = ws if ws != folder and _is_under(ws, folder) else None
    out = []
    for p in sorted(folder.rglob("*")):
        if skip is not None and _is_under(p, skip):
            continue
        if p.is_file() and p.suffix.lower() in exts and not sc.is_sidecar(p):
            if any(Path(str(p) + t).exists() for t in temps):
                continue
            out.append(p)
    return out


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def plan_process(cfg: Config, enrichers: list[Enricher] | None = None,
                 parser: Parser | None = None) -> tuple[Plan, Counter]:
    layout = cfg.layout
    naming = cfg.naming
    parser = parser or make_parser(cfg)
    enrichers = enrichers or []
    plan = Plan(kind="process")
    counts: Counter = Counter()
    taken: set[str] = set()

    for root in (layout.inbox, layout.staging):
        for video in _videos(root, cfg):
            prev = sc.read(video)
            if prev and prev.get("original_folders") is not None:
                folders = prev["original_folders"]
                original_name = prev.get("original_name", video.name)
            else:
                folders = list(video.parent.relative_to(root).parts)
                original_name = video.name
            parsed = parser.parse(original_name, folders)
            info = info_from_parsed(parsed)
            data = dict(prev or {})
            data.update({
                "version": 1,
                "original_name": original_name,
                "original_folders": folders,
                "parsed": parsed.to_dict(),
            })
            retry, notes = False, []
            for enrich in enrichers:
                result = enrich(parsed, data, path=video)
                if result is None:
                    continue
                info = merge_info(info, result.info)
                data.update(result.data)
                retry = retry or result.retry
                if result.note:
                    notes.append(result.note)
            # 人工或 AI 已確認過的欄位優先
            info = merge_info(info, data.get("manual") or {})
            identified = parsed.identified or bool(info.get("author") or info.get("characters"))
            if retry:
                stage, dest_dir = "staging", layout.staging
            elif identified:
                stage, dest_dir = "refine", layout.refine
            else:
                stage, dest_dir = "identify", layout.identify
            data["info"] = {**info, "characters": parser.dic.display_characters(info.get("characters") or [])}
            info = data["info"]
            data["stage"] = stage
            data.setdefault("history", []).append(
                {"t": datetime.now().isoformat(timespec="seconds"), "step": "process", "stage": stage})

            name = build_final_name(info, video.suffix, naming["illegal_char_style"],
                                    naming["max_name_length"])
            dst = unique_path(dest_dir / name, taken)
            if dst == video:
                continue
            counts[stage] += 1
            plan.ops.append(Op(action="move", src=str(video), dst=str(dst),
                               note="；".join([f"類型：{_type_label(parsed)}", *notes]),
                               sidecar_mode="write", sidecar=data, sidecar_prev=prev))
    return plan, counts


def _type_label(p: Parsed) -> str:
    from vsort.parser import TYPE_LABEL
    label = TYPE_LABEL[p.name_type]
    if p.site:
        label += f"（{p.site}{' ' + p.source_id if p.source_id else ''}）"
    return label


STAGE_LABEL = {"refine": "03 AI 待補標", "identify": "03 AI 待辨識", "staging": "02 待重新查詢"}
