"""一鍵分類（網頁首頁用）：挑選的檔案或資料夾 → 01_Inbox → 解析與來源查詢 → AI 判定 → 04／05。

和命令列一樣先試跑再執行（F10）：
- 放進 01 之前，先列出會搬／複製哪些檔案、略過哪些，等你在網頁按「確認」。
- AI 判定之後，先列出每部的結果檔名、去 04 還是 05 與原因，你可以把任何一部勾成「改送人工確認」，
  按「確認歸檔」才真的搬；按「取消」檔案留在 03，之後可以從首頁接著跑。
每一步都走同一套可還原的紀錄。一次只跑一件；模型載入一次後保留。
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path

from vsort.config import Config
from vsort.journal import Run, execute, incomplete_runs, save_plan
from vsort.naming import unique_path
from vsort.process import _videos


class Cancelled(Exception):
    pass


class Job:
    """一次分類工作的狀態：idle／running／waiting（等你確認）／done／cancelled／error。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"
        self.step = ""
        self.log: list[str] = []
        self.counts: dict = {}
        self.pending: dict | None = None  # 等你確認的試跑清單
        self.started = ""
        self.ended = ""
        self.error = ""
        self.batch = ""
        self._event = threading.Event()
        self._decision: dict | None = None
        self._thread: threading.Thread | None = None
        self._analyzer = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "step": self.step, "log": self.log[-200:], "counts": self.counts, "batch": self.batch,
                    "pending": self.pending, "started": self.started, "ended": self.ended, "error": self.error}

    def say(self, msg: str) -> None:
        with self.lock:
            self.log.append(f"{datetime.now():%H:%M:%S} {msg}")

    @property
    def busy(self) -> bool:
        return self.state in ("running", "waiting")

    def start(self, cfg: Config, paths: list[Path], copy: bool | None = None, analyzer=None) -> None:
        if self.busy:
            raise RuntimeError("已經有一件分類在跑，等它跑完（或先確認／取消）再開始下一件。")
        if incomplete_runs(cfg.layout):
            raise RuntimeError("有上次中斷、還沒做完的紀錄，請先在命令列執行 vsort resume。")
        if not paths and not pending_in_workspace(cfg):
            raise RuntimeError("請先勾選要分類的影片或資料夾")
        from vsort.batches import new_batch_id
        with self.lock:
            self.state, self.step, self.log, self.counts, self.error, self.pending = "running", "開始", [], {}, "", None
            self.started, self.ended = datetime.now().isoformat(timespec="seconds"), ""
            self.batch = new_batch_id() if paths else ""
        self._event.clear()
        self._decision = None
        if analyzer is not None:
            self._analyzer = analyzer
        self._thread = threading.Thread(target=self._run, args=(cfg, paths, copy), daemon=True)
        self._thread.start()

    def start_rematch(self, cfg: Config) -> None:
        """首頁按鈕：重新比對未知作者（在背景跑）。"""
        if self.busy:
            raise RuntimeError("已經有一件工作在跑，等它跑完再比對。")
        if incomplete_runs(cfg.layout):
            raise RuntimeError("有上次中斷、還沒做完的紀錄，請先在命令列執行 vsort resume。")
        with self.lock:
            self.state, self.step, self.log, self.counts, self.error, self.pending = "running", "重新比對未知作者", [], {}, "", None
            self.started, self.ended = datetime.now().isoformat(timespec="seconds"), ""
            self.batch = ""

        def go():
            try:
                n = rematch_unknown(cfg, self)
                with self.lock:
                    self.state, self.step, self.counts = "done", "比對完成", {"rematched": n}
                self.say(f"重新比對完成：{n} 部送人工確認" if n else "重新比對完成：沒有夠像的，作者未知 維持原樣")
            except Exception as exc:
                with self.lock:
                    self.state, self.error = "error", f"{exc}"
                    self.log.append(traceback.format_exc().strip().splitlines()[-1])
            finally:
                with self.lock:
                    self.ended = datetime.now().isoformat(timespec="seconds")

        self._thread = threading.Thread(target=go, daemon=True)
        self._thread.start()

    def start_undo(self, cfg: Config, batch: str) -> None:
        """整批還原（在背景跑，進度一樣寫進 log）。"""
        if self.busy:
            raise RuntimeError("已經有一件工作在跑，等它跑完再還原。")
        if incomplete_runs(cfg.layout):
            raise RuntimeError("有上次中斷、還沒做完的紀錄，請先在命令列執行 vsort resume。")
        with self.lock:
            self.state, self.step, self.log, self.counts, self.error, self.pending = "running", f"整批還原 {batch}", [], {}, "", None
            self.started, self.ended = datetime.now().isoformat(timespec="seconds"), ""
            self.batch = batch

        def go():
            from vsort.batches import undo_batch
            try:
                runs = undo_batch(cfg, batch, progress=lambda i, n, op, err: self.say(
                    f"  {'✗' if err else '✓'} {i}/{n} {Path(op.dst).name}" + (f" ← {err}" if err else "")))
                n_ok = sum(len(r.done) for r in runs)
                n_fail = sum(len(r.failed) for r in runs)
                self.say(f"整批還原完成：{len(runs)} 份紀錄，{n_ok} 筆成功，{n_fail} 筆失敗")
                with self.lock:
                    self.state, self.step = ("done" if not n_fail else "error"), "還原完成"
                    self.counts = {"restored": n_ok, "failed": n_fail}
                    if n_fail:
                        self.error = f"{n_fail} 筆還原失敗，詳見紀錄"
            except Exception as exc:
                with self.lock:
                    self.state, self.error = "error", f"{exc}"
                    self.log.append(traceback.format_exc().strip().splitlines()[-1])
            finally:
                with self.lock:
                    self.ended = datetime.now().isoformat(timespec="seconds")

        self._thread = threading.Thread(target=go, daemon=True)
        self._thread.start()

    def answer(self, decision: dict) -> None:
        """網頁按了確認或取消。"""
        if self.state != "waiting":
            raise RuntimeError("現在沒有等待確認的清單")
        self._decision = decision
        with self.lock:
            self.state, self.pending = "running", None
        self._event.set()

    def ask(self, kind: str, rows: list[dict], summary: dict) -> dict:
        """工作執行緒：把試跑清單交給網頁，等到按下確認或取消。"""
        with self.lock:
            self.state = "waiting"
            self.pending = {"kind": kind, "rows": rows, **summary}
        self._event.clear()
        self.say("等你在網頁上確認…")
        self._event.wait()
        d = self._decision or {}
        if not d.get("accept"):
            raise Cancelled(kind)
        return d

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run(self, cfg: Config, paths: list[Path], copy: bool | None) -> None:
        try:
            run_all(cfg, paths, copy, self)
            with self.lock:
                self.state, self.step = "done", "完成"
        except Cancelled as c:
            with self.lock:
                self.state, self.step = "cancelled", "已取消"
            self.say("已取消：" + ("沒有動任何檔案。" if str(c) == "ingest"
                                else "影片留在 03_AI_Workspace，之後按「開始分類」（不用挑選）就會接著判定。"))
        except Exception as exc:  # 把錯誤顯示在網頁上，不讓伺服器掛掉
            with self.lock:
                self.state, self.error = "error", f"{exc}"
                self.log.append(traceback.format_exc().strip().splitlines()[-1])
        finally:
            with self.lock:
                self.ended = datetime.now().isoformat(timespec="seconds")
                self.pending = None

    def _set_step(self, name: str) -> None:
        with self.lock:
            self.step = name

    def analyzer(self, cfg: Config):
        if self._analyzer is None:
            from vsort.analyze import Analyzer, options
            self._analyzer = Analyzer(cfg, options(cfg))
        return self._analyzer


def pending_in_workspace(cfg: Config) -> int:
    lay = cfg.layout
    return sum(len(_videos(d, cfg)) for d in (lay.inbox, lay.staging, lay.refine, lay.identify))


def _execute(cfg: Config, plan, job: Job, label: str, **extra) -> Run | None:
    if not plan.ops:
        job.say(f"{label}：沒有要執行的動作")
        return None
    save_plan(plan, cfg.layout)
    run = Run.create(cfg.layout, plan.kind, plan.ops, plan_id=plan.plan_id, **extra)
    plan.applied_run = run.run_id
    save_plan(plan, cfg.layout)
    total = len(plan.ops)

    def progress(i, n, op, err):
        if err:
            job.say(f"  ✗ {i}/{n} {Path(op.dst).name} ← {err}")
        elif i % 5 == 0 or i == n:
            job.say(f"  {i}/{n} {Path(op.dst).name}")

    execute(run, progress)
    job.say(f"{label}：完成 {len(run.done)}/{total}，失敗 {len(run.failed)}（紀錄 {run.run_id}，可用 vsort undo 還原）")
    return run


def _short(path: str, cfg: Config) -> str:
    p = Path(path)
    for base, label in ((cfg.layout.library, "04"), (cfg.layout.review, "05"), (cfg.layout.workspace, "工作區")):
        try:
            return f"[{label}] " + p.relative_to(base).as_posix()
        except ValueError:
            pass
    return path


def send_to_review(plan, cfg: Config, srcs: set[str]) -> None:
    """試跑清單裡你勾成「改送人工確認」的影片：目的地改成 05，並記下原因。"""
    lay = cfg.layout
    taken = {op.dst.lower() for op in plan.ops}
    for op in plan.ops:
        if op.src not in srcs or (op.sidecar or {}).get("stage") != "library":
            continue
        new = unique_path(lay.review / Path(op.dst).name, taken)
        op.dst = str(new)
        op.certainty = "可能誤判"
        sc = dict(op.sidecar or {})
        sc["stage"], sc["certainty"] = "review", "可能誤判"
        sc["reasons"] = list(sc.get("reasons") or []) + ["你在試跑清單勾選了人工確認"]
        op.sidecar = sc
        op.note = "；".join(sc["reasons"])


def run_all(cfg: Config, paths: list[Path], copy: bool | None, job: Job) -> dict:
    """依序執行三步（兩個確認點）；回傳各階段數量。"""
    from vsort.ingest import plan_ingest
    from vsort.process import make_parser, plan_process
    from vsort.analyze import make_searcher, plan_analyze
    from vsort.review import export

    cfg.layout.ensure()
    if paths:
        mode = ("copy" if copy else "move") if copy is not None else None
        job._set_step("1/3 放進 01_Inbox（試跑）")
        plan = plan_ingest(paths, cfg, mode=mode, batch=job.batch)
        action = "複製" if plan.meta.get("mode") == "copy" else "搬移"
        job.say(f"試跑：{action} {len(plan.ops)} 部影片到 01_Inbox，略過 {len(plan.skipped)} 個")
        rows = [{"src": op.src, "dst": _short(op.dst, cfg), "note": op.note} for op in plan.ops]
        job.ask("ingest", rows, {"action": action, "skipped": [[p, r] for p, r in plan.skipped[:200]],
                                 "non_video": plan.meta.get("non_video_count", 0)})
        job._set_step("1/3 放進 01_Inbox")
        _execute(cfg, plan, job, f"{action}到 01_Inbox", batch=job.batch)
        job.say(f"批次編號 {job.batch}（之後可以在首頁「歷史批次」整批還原）")
    else:
        job.say(f"沒有新挑選的影片，接著處理工作區裡還沒判定完的 {pending_in_workspace(cfg)} 部")

    job._set_step("2/3 解析檔名與查來源")
    enrichers = []
    if (cfg.data.get("lookup") or {}).get("enabled", True):
        from vsort.lookup import build_enrichers
        enrichers = build_enrichers(cfg, make_parser(cfg))
    job.say("解析檔名、查來源網站（第一次查一位作者可能要 1～3 分鐘）…")
    plan, counts = plan_process(cfg, enrichers=enrichers)
    run = _execute(cfg, plan, job, "解析與分流", **({"batch": job.batch} if job.batch else {}))
    if run is not None:
        try:
            from vsort.authors import update_authors
            added, queued = update_authors(cfg)
            if added or queued:
                job.say(f"作者別名表：新增 {added} 位，{queued} 筆疑似別名等你確認（vsort authors）")
        except Exception as exc:
            job.say(f"作者別名表更新失敗（不影響分類）：{exc}")

    job._set_step("3/3 AI 判定（試跑）")
    job.say("AI 分析中（第一次要載入模型約 10～20 秒；每部約 5～10 秒）…")
    analyzer = job.analyzer(cfg)
    plan, counts = plan_analyze(cfg, analyzer, progress=lambda name: job.say(f"  分析 {name[:60]}"),
                                searcher=make_searcher(cfg))
    rows = []
    for op in plan.ops:
        scd = op.sidecar or {}
        rows.append({"src": op.src, "name": Path(op.src).name, "dst_name": Path(op.dst).name,
                     "to": "library" if scd.get("stage") == "library" else "review",
                     "folder": Path(op.dst).parent.name if scd.get("stage") == "library" else "05_Review",
                     "reasons": scd.get("reasons") or [], "info": scd.get("info") or {}})
    job.say(f"試跑：已確定 {counts.get('library', 0)} 部、可能誤判 {counts.get('review', 0)} 部")
    decision = job.ask("analyze", rows, {"library": counts.get("library", 0), "review": counts.get("review", 0)})
    to_review = set(decision.get("to_review") or [])
    if to_review:
        send_to_review(plan, cfg, to_review)
        job.say(f"你把 {len(to_review)} 部改送人工確認")
    job._set_step("3/3 AI 判定與歸檔")
    _execute(cfg, plan, job, "AI 判定與歸檔", **({"batch": job.batch} if job.batch else {}))
    export(cfg)
    result = {"library": sum(1 for op in plan.ops if op.certainty == "已確定"),
              "review": sum(1 for op in plan.ops if op.certainty != "已確定")}
    # 未知作者的影片：已知作者變多了，重新用畫風指紋比對一次，夠像的送 05 預填作者
    try:
        result["rematched"] = rematch_unknown(cfg, job)
    except Exception as exc:
        job.say(f"重新比對未知作者失敗（不影響這批）：{exc}")
    with job.lock:
        job.counts = result
    job.say(f"完成：已確定 → 04 收藏庫 {result['library']} 部；可能誤判 → 05 人工確認 {result['review']} 部")
    return result


def rematch_unknown(cfg: Config, job: Job) -> int:
    """作者未知 裡的影片和現有作者比畫風，夠像的搬到 05 並預填作者。回傳搬了幾部。"""
    from vsort.rematch import plan_rematch
    from vsort.review import export
    plan, n = plan_rematch(cfg)
    if not n:
        return 0
    job.say(f"重新比對 作者未知 的 {n} 部：{len(plan.ops)} 部找到相似的作者，送人工確認")
    if plan.ops:
        _execute(cfg, plan, job, "未知作者重新比對")
        export(cfg)
    return len(plan.ops)
