"""試跑計畫（F10）、操作紀錄與還原（F11）、中斷續跑（F15）。

不用資料庫：
- 試跑時把「打算怎麼動」存成 _reports/plan-*.json 與同名 .csv（Excel 可開）。
- 實際執行時每個動作都寫進 _logs/*.jsonl（純文字，一行一筆），先寫「開始」、做完再寫「完成」，
  每筆都立即寫入磁碟。中斷後依紀錄判斷每個檔案的實際狀態，從原處繼續。
- 還原就是把已完成的動作反向做一次，還原本身也是一份可續跑、可再還原的紀錄。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from vsort.config import Layout
from vsort import sidecar as sc
from vsort.fileops import copy_verified, files_match, move_file, part_path

OP_FIELDS = ("action", "src", "dst", "certainty", "note", "sidecar_mode", "sidecar", "sidecar_prev")
ACTION_LABEL = {"move": "搬移", "copy": "複製", "remove_copy": "刪除複本"}


@dataclass
class Op:
    action: str  # move | copy | remove_copy
    src: str
    dst: str
    certainty: str = ""  # 已確定 / 可能誤判
    note: str = ""
    # 影片旁的資訊檔：""＝原樣跟著搬；write＝搬完寫入 sidecar；delete＝搬完刪除
    sidecar_mode: str = ""
    sidecar: dict | None = None
    sidecar_prev: dict | None = None  # 搬移前的內容，還原用


@dataclass
class Plan:
    kind: str
    ops: list[Op] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    plan_id: str = ""
    created: str = ""
    applied_run: str | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")  # 同一秒內建立的多份紀錄也要排得出先後


def _new_id(folder: Path, prefix: str, ext: str) -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate, n = base, 1
    while (folder / f"{prefix}{candidate}{ext}").exists():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


# ---------- 試跑計畫 ----------

def save_plan(plan: Plan, layout: Layout) -> tuple[Path, Path]:
    layout.reports.mkdir(parents=True, exist_ok=True)
    if not plan.plan_id:
        plan.plan_id = _new_id(layout.reports, "plan-", ".json")
        plan.created = _now()
    jpath = layout.reports / f"plan-{plan.plan_id}.json"
    cpath = layout.reports / f"plan-{plan.plan_id}.csv"
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(asdict(plan), fh, ensure_ascii=False, indent=1)
    with open(cpath, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["狀態", "判定", "動作", "原路徑", "新路徑", "說明"])
        for op in plan.ops:
            w.writerow(["會執行", op.certainty, ACTION_LABEL.get(op.action, op.action),
                        op.src, op.dst, op.note])
        for path, reason in plan.skipped:
            w.writerow(["略過", "", "", path, "", reason])
    return jpath, cpath


def load_plan(layout: Layout, plan_id: str | None = None) -> Plan:
    if plan_id:
        jpath = layout.reports / f"plan-{plan_id}.json"
    else:
        plans = sorted(layout.reports.glob("plan-*.json"))
        if not plans:
            raise FileNotFoundError("還沒有任何試跑計畫，請先試跑一次。")
        jpath = plans[-1]
    with open(jpath, encoding="utf-8") as fh:
        d = json.load(fh)
    d["ops"] = [Op(**o) for o in d["ops"]]
    d["skipped"] = [tuple(s) for s in d["skipped"]]
    return Plan(**d)


# ---------- 執行紀錄 ----------

class Run:
    def __init__(self, path: Path):
        self.path = path
        self.run_id = path.stem
        self.header: dict = {}
        self.ops: list[Op] = []
        self.begun: set[int] = set()
        self.done: set[int] = set()
        self.failed: dict[int, str] = {}
        self.ended: dict | None = None
        self.undone_by: dict | None = None
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 斷電時最後一行可能寫一半，忽略
                t = rec.get("type")
                if t == "header":
                    self.header = rec
                elif t == "op":
                    self.ops.append(Op(**{k: rec[k] for k in OP_FIELDS if k in rec}))
                elif t == "begin":
                    self.begun.add(rec["i"])
                elif t == "done":
                    self.done.add(rec["i"])
                    self.failed.pop(rec["i"], None)
                elif t == "fail":
                    self.failed[rec["i"]] = rec.get("error", "")
                elif t == "end":
                    self.ended = rec
                elif t == "resume":
                    self.ended = None
                elif t == "undone_by":
                    self.undone_by = rec

    def write(self, rec: dict) -> None:
        rec.setdefault("t", _now())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @classmethod
    def create(cls, layout: Layout, kind: str, ops: list[Op], **extra) -> "Run":
        layout.logs.mkdir(parents=True, exist_ok=True)
        rid = _new_id(layout.logs, f"{kind}-", ".jsonl")
        run = cls(layout.logs / f"{kind}-{rid}.jsonl")
        run.write({"type": "header", "kind": kind, "version": 1, **extra})
        run.header = {"kind": kind, **extra}
        for i, op in enumerate(ops):
            run.write({"type": "op", "i": i, **asdict(op)})
        run.ops = list(ops)
        return run

    @property
    def kind(self) -> str:
        return self.header.get("kind", "?")

    @property
    def complete(self) -> bool:
        return self.ended is not None

    @property
    def created(self) -> str:
        return self.header.get("t", "")


def list_runs(layout: Layout) -> list[Run]:
    if not layout.logs.exists():
        return []
    runs = [Run(p) for p in layout.logs.glob("*.jsonl")]
    return sorted(runs, key=lambda r: (r.created, r.run_id))


def incomplete_runs(layout: Layout) -> list[Run]:
    return [r for r in list_runs(layout) if not r.complete]


# ---------- 執行 ----------

def _precheck(op: Op) -> str | None:
    src, dst = Path(op.src), Path(op.dst)
    if op.action == "remove_copy":
        if dst.exists() and not src.exists():
            return "原檔已不在，為了安全不刪除複本"
        return None
    if not src.exists():
        return "來源檔案不在了（可能已被之後的操作移走）"
    if dst.exists():
        return "目的地已有同名檔案，不覆蓋"
    return None


def _perform(op: Op) -> None:
    src, dst = Path(op.src), Path(op.dst)
    if op.action == "move":
        move_file(src, dst)
        _finish_sidecar(op)
    elif op.action == "copy":
        copy_verified(src, dst)
    elif op.action == "remove_copy":
        if dst.exists():
            if not files_match(src, dst):
                raise OSError("複本和原檔內容不同，不刪除")
            os.remove(dst)
    else:
        raise ValueError(f"未知動作：{op.action}")


def _finish_sidecar(op: Op) -> None:
    """影片搬完後處理資訊檔；可重複執行。"""
    src, dst = Path(op.src), Path(op.dst)
    for src_x, dst_x in ((sc.path_for(src), sc.path_for(dst)), (sc.preview_for(src), sc.preview_for(dst))):
        if src_x.exists():
            if dst_x.exists():
                src_x.unlink()
            else:
                move_file(src_x, dst_x)
    dst_sc, dst_prev = sc.path_for(dst), sc.preview_for(dst)
    if op.sidecar_mode == "write" and op.sidecar is not None:
        sc.write_at(dst_sc, op.sidecar)
        preview = op.sidecar.get("preview")
        if preview and Path(preview).exists() and op.sidecar.get("stage") == "review":
            shutil.copyfile(preview, dst_prev)  # 送人工確認時附上預覽圖
        elif dst_prev.exists() and op.sidecar.get("stage") != "review":
            dst_prev.unlink()
    elif op.sidecar_mode == "delete":
        for x in (dst_sc, dst_prev):
            if x.exists():
                x.unlink()


def _reconcile(op: Op) -> str:
    """處理「已開始但沒寫完成」的動作。回傳 done / redo，或拋出錯誤。"""
    src, dst = Path(op.src), Path(op.dst)
    part = part_path(dst)
    if op.action == "move":
        if src.exists() and dst.exists():
            # 跨硬碟搬移：複本已核對並改名，只差刪除原檔
            if files_match(src, dst):
                os.remove(src)
                return "done"
            raise OSError("來源和目的地都有檔案且內容不同，請人工檢查")
        if dst.exists():
            return "done"
        if src.exists():
            if part.exists():
                part.unlink()
            return "redo"
        raise OSError("來源和目的地都找不到檔案")
    if op.action == "copy":
        if dst.exists():
            return "done"
        if part.exists():
            part.unlink()
        return "redo"
    if op.action == "remove_copy":
        return "done" if not dst.exists() else "redo"
    raise ValueError(op.action)


def execute(run: Run, progress=None) -> Run:
    """執行（或續跑）一份紀錄裡尚未完成的動作。"""
    total = len(run.ops)
    for i, op in enumerate(run.ops):
        if i in run.done:
            continue
        try:
            if i in run.begun:
                state = _reconcile(op)
                if state == "done":
                    if op.action == "move":
                        _finish_sidecar(op)
                    run.write({"type": "done", "i": i, "reconciled": True})
                    run.done.add(i)
                    continue
            else:
                problem = _precheck(op)
                if problem:
                    raise OSError(problem)
                run.write({"type": "begin", "i": i})
                run.begun.add(i)
            _perform(op)
            run.write({"type": "done", "i": i})
            run.done.add(i)
            run.failed.pop(i, None)
        except OSError as exc:
            msg = str(exc) or exc.__class__.__name__
            run.write({"type": "fail", "i": i, "error": msg})
            run.failed[i] = msg
        if progress:
            progress(i + 1, total, op, run.failed.get(i))
    status = "completed" if not run.failed else "completed_with_errors"
    run.write({"type": "end", "status": status, "done": len(run.done), "failed": len(run.failed)})
    run.ended = {"status": status}
    _after_end(run)
    return run


def resume(run: Run, progress=None) -> Run:
    run.write({"type": "resume"})
    run.ended = None
    return execute(run, progress)


def _after_end(run: Run) -> None:
    target = run.header.get("undo_of")
    if target and not run.failed and not run.header.get("partial"):
        orig = Path(run.path.parent / f"{target}.jsonl")
        if orig.exists():
            Run(orig).write({"type": "undone_by", "run": run.run_id})


# ---------- 還原 ----------

def pick_undo_target(layout: Layout, run_id: str | None = None) -> Run:
    if run_id:
        path = layout.logs / f"{run_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"找不到紀錄：{run_id}")
        return Run(path)
    for run in reversed(list_runs(layout)):
        if run.kind != "undo" and run.undone_by is None and run.done:
            return run
    raise FileNotFoundError("沒有可以還原的紀錄。")


def build_undo_plan(run: Run, only: set[int] | None = None, check: bool = True) -> Plan:
    """only：只還原這些動作（整批還原時，一份紀錄可能混了不同批次的影片）。
    check=False：不檢查檔案現在在不在（整批還原的試跑清單用：前面的步驟退回後檔案才會回到位）。"""
    if not run.complete:
        raise RuntimeError(f"紀錄 {run.run_id} 還沒執行完，請先續跑（vsort resume）再還原。")
    plan = Plan(kind="undo", meta={"undo_of": run.run_id})
    for i in sorted(run.done, reverse=True):
        if only is not None and i not in only:
            continue
        op = run.ops[i]
        if op.action == "move":
            mode, data = "", None
            if op.sidecar_mode in ("write", "delete"):
                mode, data = ("write", op.sidecar_prev) if op.sidecar_prev is not None else ("delete", None)
            rev = Op("move", src=op.dst, dst=op.src, note=f"還原 {run.run_id}",
                     sidecar_mode=mode, sidecar=data)
        elif op.action == "copy":
            rev = Op("remove_copy", src=op.src, dst=op.dst, note=f"還原 {run.run_id}",
                     sidecar_mode="delete" if op.sidecar_mode == "write" else "")
        elif op.action == "remove_copy":
            rev = Op("copy", src=op.src, dst=op.dst, note=f"還原 {run.run_id}")
        else:
            continue
        problem = _precheck(rev) if check else None
        if problem:
            plan.skipped.append((rev.src, problem))
        else:
            plan.ops.append(rev)
    return plan
