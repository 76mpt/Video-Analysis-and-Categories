"""把使用者挑選的影片放進 01_Inbox 的計畫（F29 + F02 + F12）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vsort.config import Config
from vsort.dupes import workspace_index
from vsort.journal import Op, Plan
from vsort.naming import fit_path, sanitize_dirname, sanitize_filename, unique_path
from vsort.scanner import scan


def plan_ingest(inputs: list[Path], cfg: Config, mode: str | None = None,
                recursive: bool = True, skip_duplicates: bool = True, batch: str | None = None) -> Plan:
    """batch：這一批的編號（沒給就新產生），寫進每部影片的資訊檔，之後可以整批還原（vsort undo --batch）。"""
    from vsort.batches import new_batch_id
    batch = batch or new_batch_id()
    layout = cfg.layout
    naming = cfg.naming
    style, max_len, max_path = (naming["illegal_char_style"], naming["max_name_length"],
                                naming["max_path_length"])
    mode = mode or cfg.data["ingest"]["mode"]
    keep_tree = cfg.data["ingest"]["keep_folder_structure"]

    result = scan(inputs, cfg, recursive=recursive, exclude=[layout.workspace, layout.library])
    plan = Plan(kind="ingest", meta={"inputs": [str(p) for p in inputs], "mode": mode, "batch": batch,
                                     "non_video_count": result.non_video_count})
    plan.skipped = [(str(p), reason) for p, reason in result.skipped]
    taken: set[str] = set()
    dup_index = workspace_index(cfg) if skip_duplicates else None

    for sel in result.files:
        if dup_index is not None:
            same = dup_index.find(sel.path)
            if same is not None:
                plan.skipped.append((str(sel.path), f"和 {same} 內容相同（重複下載），略過"))
                continue
            dup_index.add(sel.path)  # 這次挑選的檔案彼此之間也要比
        dest_dir = layout.inbox
        if keep_tree and sel.root is not None:
            rel_parts = [sel.root.name, *sel.path.parent.relative_to(sel.root).parts]
            for part in rel_parts:
                dest_dir = dest_dir / sanitize_dirname(part, style)
        name = sanitize_filename(sel.path.name, style, max_len)
        dst = unique_path(fit_path(dest_dir / name, max_path), taken)
        notes = []
        if dst.name != sel.path.name:
            notes.append("檔名已調整（非法字元、長度或同名）")
        side = {"version": 1, "original_path": str(sel.path), "batch": batch,
                "history": [{"t": datetime.now().isoformat(timespec="seconds"), "step": "ingest", "stage": "inbox",
                             "mode": mode}]}
        plan.ops.append(Op(action=mode, src=str(sel.path), dst=str(dst), note="；".join(notes),
                           sidecar_mode="write", sidecar=side))
    return plan
