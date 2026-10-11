"""批次（一次從首頁或 vsort ingest 開始的那一批影片）的紀錄與整批還原。

- 批次編號在放進 01 時產生（batch-日期-時間），寫進每部影片的資訊檔（`batch`）和執行紀錄的標頭；
  之後的解析分流、AI 歸檔、人工確認都跟著資訊檔帶著這個編號。
- 整批還原：找出所有動過這批影片的紀錄（含人工確認），依相反順序各自建立還原計畫並執行，
  搬移的搬回原位置、複製的刪掉複本。每一份還原本身也是紀錄，可以再還原。
- 不同批次互不影響；同一部影片若被後面的批次再動過，那幾筆會被略過並列在清單裡。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from vsort import sidecar as sc
from vsort.config import Config
from vsort.journal import Op, Plan, Run, build_undo_plan, execute, list_runs, save_plan
from vsort.process import _videos


def new_batch_id() -> str:
    return f"batch-{datetime.now():%Y%m%d-%H%M%S}"


def op_batch(op: Op) -> str | None:
    return (op.sidecar or {}).get("batch") or None


def run_batch_indexes(run: Run, batch: str) -> set[int]:
    """這份紀錄裡屬於某批次、而且已完成的動作。"""
    if run.kind == "undo":
        return set()
    if run.header.get("batch") == batch:
        return set(run.done)
    return {i for i in run.done if op_batch(run.ops[i]) == batch}


def batch_runs(cfg: Config, batch: str) -> list[tuple[Run, set[int]]]:
    """動過這批影片、還沒被還原的紀錄（依時間順序）。"""
    out = []
    for run in list_runs(cfg.layout):
        if run.undone_by or not run.complete:
            continue
        idx = run_batch_indexes(run, batch)
        if idx:
            out.append((run, idx))
    return out


def where_now(cfg: Config) -> dict[str, Counter]:
    """每個批次的影片現在各在哪個階段（掃工作區與收藏庫的資訊檔）。"""
    lay = cfg.layout
    stages = [("01", lay.inbox), ("02", lay.staging), ("03", lay.identify), ("03", lay.refine),
              ("04", lay.library), ("05", lay.review)]
    out: dict[str, Counter] = defaultdict(Counter)
    for label, folder in stages:
        for v in _videos(folder, cfg):
            b = (sc.read(v) or {}).get("batch")
            if b:
                out[b][label] += 1
    return out


def list_batches(cfg: Config) -> list[dict]:
    """歷史批次：編號、時間、來源、部數、各紀錄、現在各在哪。最新的在前。"""
    info: dict[str, dict] = {}
    for run in list_runs(cfg.layout):
        if run.kind == "undo":
            continue
        batches = {run.header.get("batch")} if run.header.get("batch") else {op_batch(op) for op in run.ops}
        for b in batches:
            if not b:
                continue
            d = info.setdefault(b, {"id": b, "created": run.created, "runs": [], "videos": 0, "sources": Counter(),
                                    "mode": ""})
            idx = run_batch_indexes(run, b)
            d["runs"].append({"run": run.run_id, "kind": run.kind, "n": len(idx), "undone": bool(run.undone_by)})
            if run.kind == "ingest":
                d["videos"] += len(idx)
                d["mode"] = run.ops[0].action if run.ops else ""
                for i in idx:
                    d["sources"][str(Path(run.ops[i].src).parent)] += 1
    now = where_now(cfg)
    out = []
    for b, d in info.items():
        d["sources"] = [s for s, _ in d["sources"].most_common(5)]
        d["now"] = dict(now.get(b, {}))
        d["undone"] = all(r["undone"] for r in d["runs"]) if d["runs"] else False
        d["active_runs"] = sum(1 for r in d["runs"] if not r["undone"])
        out.append(d)
    return sorted(out, key=lambda d: d["created"], reverse=True)


def plan_undo_batch(cfg: Config, batch: str) -> list[tuple[Run, Plan, bool]]:
    """整批還原的試跑：依相反順序，每份紀錄一個還原計畫。回傳 [(原紀錄, 還原計畫, 是否只還原部分)]。
    試跑時不檢查檔案現在在不在：要等前一步退回後，檔案才會回到下一步預期的位置。"""
    plans = []
    for run, idx in reversed(batch_runs(cfg, batch)):
        plan = build_undo_plan(run, only=idx, check=False)
        plans.append((run, plan, idx != set(run.done)))
    return plans


def undo_batch(cfg: Config, batch: str, progress=None) -> list[Run]:
    """真的整批還原。每一步在執行前才建立還原計畫（前一步退回後檔案位置才正確）；回傳還原紀錄。"""
    done = []
    for run, idx in reversed(batch_runs(cfg, batch)):
        plan = build_undo_plan(run, only=idx)
        partial = idx != set(run.done)
        if not plan.ops:
            continue
        save_plan(plan, cfg.layout)
        undo = Run.create(cfg.layout, "undo", plan.ops, undo_of=run.run_id, partial=partial or bool(plan.skipped),
                          batch=batch)
        plan.applied_run = undo.run_id
        save_plan(plan, cfg.layout)
        execute(undo, progress)
        done.append(undo)
    return done
