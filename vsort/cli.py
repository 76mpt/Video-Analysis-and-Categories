"""命令列介面。所有會動到檔案的指令預設都是試跑，加 --apply 才真的執行。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vsort import __version__
from vsort.config import Config, load_config
from vsort.ingest import plan_ingest
from vsort.process import STAGE_LABEL, make_parser, plan_process
from vsort.journal import (ACTION_LABEL, Plan, Run, build_undo_plan, execute, incomplete_runs,
                           list_runs, load_plan, pick_undo_target, resume, save_plan)


def _out(msg: str = "") -> None:
    print(msg, flush=True)


def _short(path: str, cfg: Config) -> str:
    p = Path(path)
    for base, label in ((cfg.layout.workspace, "工作區"), (cfg.layout.library, "收藏庫")):
        try:
            return f"[{label}]\\" + str(p.relative_to(base))
        except ValueError:
            pass
    return path


def _show_plan(plan: Plan, cfg: Config, limit: int = 30) -> None:
    for op in plan.ops[:limit]:
        tag = f"[{op.certainty}] " if op.certainty else ""
        _out(f"  {tag}{ACTION_LABEL.get(op.action, op.action)}  {op.src}")
        _out(f"        → {_short(op.dst, cfg)}" + (f"   （{op.note}）" if op.note else ""))
    if len(plan.ops) > limit:
        _out(f"  …其餘 {len(plan.ops) - limit} 筆請看 CSV 清單")
    if plan.skipped:
        _out(f"\n  略過 {len(plan.skipped)} 個：")
        for path, reason in plan.skipped[:limit]:
            _out(f"    {path}  ← {reason}")
    nv = plan.meta.get("non_video_count")
    if nv:
        _out(f"  （資料夾內另有 {nv} 個非影片檔案，已忽略）")


def _progress(i: int, total: int, op, err) -> None:
    mark = "✗" if err else "✓"
    _out(f"  {mark} {i}/{total} {Path(op.dst).name}" + (f"  ← {err}" if err else ""))


def _guard_incomplete(cfg: Config) -> bool:
    pending = incomplete_runs(cfg.layout)
    if pending:
        _out("有上次中斷、還沒做完的紀錄，請先執行 `vsort resume` 把它做完：")
        for r in pending:
            _out(f"  {r.run_id}（完成 {len(r.done)}/{len(r.ops)}）")
        return False
    return True


def _run_plan(plan: Plan, cfg: Config, **extra) -> int:
    if not plan.ops:
        _out("沒有要執行的動作。")
        return 0
    if not _guard_incomplete(cfg):
        return 2
    cfg.layout.ensure()
    run = Run.create(cfg.layout, plan.kind, plan.ops, plan_id=plan.plan_id, **extra)
    plan.applied_run = run.run_id
    save_plan(plan, cfg.layout)
    _out(f"開始執行，紀錄：{run.path}")
    execute(run, _progress)
    _out(f"\n完成 {len(run.done)}/{len(run.ops)}，失敗 {len(run.failed)}。紀錄編號：{run.run_id}")
    if run.failed:
        _out("失敗的項目可以處理後用 `vsort resume` 重試。")
    _out(f"要復原這次的變動：vsort undo {run.run_id} --apply")
    return 1 if run.failed else 0


def _pick_paths(folder: bool) -> list[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if folder:
        chosen = filedialog.askdirectory(title="選擇要處理的影片資料夾")
        paths = [chosen] if chosen else []
    else:
        paths = list(filedialog.askopenfilenames(title="選擇要處理的影片"))
    root.destroy()
    return [Path(p) for p in paths]


# ---------- 指令 ----------

def cmd_init(args, cfg: Config) -> int:
    created = cfg.layout.ensure()
    _out(f"工作區：{cfg.layout.workspace}")
    for label, path in cfg.layout.stage_dirs():
        _out(f"  {label}：{path}")
    _out(f"新建立 {len(created)} 個資料夾。")
    return 0


def cmd_ingest(args, cfg: Config) -> int:
    paths = [Path(p) for p in args.paths]
    if args.pick or args.pick_folder:
        paths += _pick_paths(folder=args.pick_folder)
    if not paths:
        _out("請指定要處理的影片或資料夾，或加 --pick / --pick-folder 用視窗挑選。")
        return 2
    mode = "copy" if args.copy else None
    plan = plan_ingest(paths, cfg, mode=mode, recursive=not args.no_recursive,
                       skip_duplicates=not args.keep_duplicates)
    jpath, cpath = save_plan(plan, cfg.layout)
    _out(f"{'實際執行' if args.apply else '試跑（不會動任何檔案）'}：{len(plan.ops)} 部影片要放進 01_Inbox")
    _show_plan(plan, cfg)
    _out(f"\n完整清單：{cpath}")
    _out(f"批次編號：{plan.meta['batch']}（之後可用 vsort undo --batch {plan.meta['batch']} 整批還原）")
    if args.apply:
        return _run_plan(plan, cfg, batch=plan.meta["batch"])
    _out(f"確認沒問題後執行：vsort apply {plan.plan_id}")
    return 0


def cmd_process(args, cfg: Config) -> int:
    cfg.layout.ensure()
    plan, counts = plan_process(cfg, enrichers=_enrichers(args, cfg))
    jpath, cpath = save_plan(plan, cfg.layout)
    _out(f"{'實際執行' if args.apply else '試跑（不會動任何檔案）'}：解析 {len(plan.ops)} 部影片")
    _show_plan(plan, cfg)
    _out("\n摘要：")
    for stage, label in STAGE_LABEL.items():
        if counts.get(stage):
            _out(f"  {label}：{counts[stage]} 部")
    _out(f"\n完整清單：{cpath}")
    if args.apply:
        code = _run_plan(plan, cfg)
        _after_process(cfg)
        return code
    if plan.ops:
        _out(f"確認沒問題後執行：vsort apply {plan.plan_id}")
    return 0


def _after_process(cfg: Config) -> None:
    from vsort.authors import update_authors
    added, queued = update_authors(cfg)
    if added or queued:
        _out(f"作者別名表：新增 {added} 位，{queued} 筆疑似別名等你確認（vsort authors）")


def _enrichers(args, cfg: Config) -> list:
    if getattr(args, "offline", False):
        return []
    from vsort.lookup import build_enrichers
    return build_enrichers(cfg, make_parser(cfg))


def cmd_analyze(args, cfg: Config) -> int:
    from vsort.analyze import Analyzer, make_searcher, options, plan_analyze

    cfg.layout.ensure()
    analyzer = Analyzer(cfg, options(cfg))
    _out("AI 分析中（第一次會載入模型，約 10～20 秒；分析結果會快取，實際執行時不會重算）…")
    plan, counts = plan_analyze(cfg, analyzer, include_library=args.library,
                                progress=lambda name: _out(f"  分析 {name}"), searcher=make_searcher(cfg))
    save_plan(plan, cfg.layout)
    _out(f"\n{'實際執行' if args.apply else '試跑（不會動任何檔案）'}：{len(plan.ops)} 部影片判定完成")
    _show_plan(plan, cfg)
    _out(f"\n摘要：已確定 → 04 收藏庫 {counts['library']} 部；可能誤判 → 05 人工確認 {counts['review']} 部")
    if args.apply:
        code = _run_plan(plan, cfg)
        _refresh_review(cfg)
        return code
    if plan.ops:
        _out(f"確認沒問題後執行：vsort apply {plan.plan_id}")
    return 0


def _refresh_review(cfg: Config) -> None:
    from vsort.review import export
    path, n = export(cfg)
    if n:
        _out(f"05 人工確認有 {n} 部，清單：{path}（填好後執行 vsort review --import）")


def cmd_review(args, cfg: Config) -> int:
    from vsort.review import export, plan_import

    if args.web:
        from vsort.webapp import serve
        serve(cfg, port=args.port, page="/review")
        return 0
    if not args.do_import:
        path, n = export(cfg)
        _out(f"05 人工確認區有 {n} 部影片，清單：{path}")
        _out("用 Excel 打開，修改要更正的欄位，在第一欄填 Y，存檔後執行：vsort review --import")
        return 0
    plan = plan_import(cfg)
    save_plan(plan, cfg.layout)
    _out(f"{'實際執行' if args.apply else '試跑（不會動任何檔案）'}：{len(plan.ops)} 部確認後歸檔到 04")
    _show_plan(plan, cfg)
    if args.apply:
        code = _run_plan(plan, cfg)
        export(cfg)
        return code
    if plan.ops:
        _out(f"確認沒問題後執行：vsort review --import --apply")
    return 0


def cmd_web(args, cfg: Config) -> int:
    """網頁首頁：挑影片（可複選資料夾或檔案）→ 一鍵分類 → 人工確認。"""
    from vsort.webapp import serve
    serve(cfg, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_dupes(args, cfg: Config) -> int:
    """列出重複的影片（只列清單，不刪除）。"""
    import csv
    from datetime import datetime
    from vsort.dupes import DupIndex, workspace_index

    if args.paths:
        idx = DupIndex(cfg.video_extensions)
        for p in args.paths:
            idx.add_tree(Path(p)) if Path(p).is_dir() else idx.add(Path(p))
    else:
        idx = workspace_index(cfg)
    groups = idx.groups()
    _out(f"找到 {len(groups)} 組重複的影片。")
    for g in groups[:args.limit]:
        _out("  " + "\n  = ".join(str(p) for p in g))
    if groups:
        cfg.layout.reports.mkdir(parents=True, exist_ok=True)
        out = cfg.layout.reports / f"dupes-{datetime.now():%Y%m%d-%H%M%S}.csv"
        with open(out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["組別", "大小(MB)", "路徑"])
            for i, g in enumerate(groups, 1):
                for p in g:
                    w.writerow([i, round(p.stat().st_size / 1048576, 1), str(p)])
        _out(f"完整清單：{out}（程式不會刪除任何檔案，要刪哪一份由你決定）")
    return 0


def cmd_label(args, cfg: Config) -> int:
    from vsort.labeling import build_queue, labels_path, serve
    cfg.layout.ensure()
    roots = [Path(p) for p in args.paths]
    _out("抽樣中（只讀檔名與 iwara 標籤，不搬檔案）…")
    queue = build_queue(cfg, roots, args.n, per_folder=args.per_folder, seed=args.seed,
                        use_lookup=not args.offline, uncertain=args.uncertain, progress=_out,
                        include_labeled=args.include_labeled.split(",") if args.include_labeled else None,
                        category_uncertain=args.category_uncertain)
    _out(f"這次要標 {len(queue)} 部，標好的會存到 {labels_path(cfg)}")
    serve(cfg, queue, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_train_style(args, cfg: Config) -> int:
    from vsort.style_train import evaluate, train
    feats = tuple(args.features.split(","))
    _out("比較動畫類型的判斷方式（5 折交叉驗證，第一次會計算特徵，比較久）…")
    rep = evaluate(cfg, features=feats, progress=_out)
    _out("")
    from vsort.naming import ANIM_DISPLAY, ANIM_ORDER
    present = [a for a in ANIM_ORDER if rep["counts"].get(a)]
    _out(f"{'方法':<24}{'完全正確':>8}{'寬鬆正確':>8}{'第一名正確':>10}  F1（{'／'.join(ANIM_DISPLAY.get(a, a) for a in present)}）")
    best = None
    for name, m in rep["methods"].items():
        f1 = "／".join("-" if m.get(f"F1_{a}") is None else f"{m[f'F1_{a}']:.2f}" for a in present)
        _out(f"{name:<24}{m['exact']:>8.0%}{m.get('lenient', 0):>8.0%}{m.get('top1', 0):>10.0%}  {f1}")
        if name.startswith("小分類器") and (best is None or m["exact"] > best[1]["exact"]):
            best = (name, m)
    cfg.layout.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.layout.reports / "style_eval.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _out(f"\n詳細結果：{out}")
    if best and not args.no_train:
        feat = best[0][best[0].index("（") + 1:best[0].index(" 特徵")] if "全部" not in best[0] else "+".join(feats)
        path = train(cfg, feat, best[1]["C"])
        _out(f"已用全部樣本訓練最好的小分類器（{best[0]}），存到 {path}")
        _out("要改用它：把 config.yaml 的 ai.style_model 改成 trained")
    return 0


def cmd_train_category(args, cfg: Config) -> int:
    from vsort.analyze import Analyzer, options
    from vsort.category_train import (CATEGORY_CLASSIFIER, FALLBACK_CLASSIFIER, FALLBACK_FEATURE_SET,
                                      evaluate_and_train)
    opts = options(cfg) | {"category_model": "qwen3vl"}
    _out("訓練類別分類器（畫面標籤＋Qwen3-VL），沒分析過的影片會先跑一次 AI，比較久…")
    rep = evaluate_and_train(cfg, Analyzer(cfg, opts), progress=_out, sure_prob=opts.get("category_sure_prob", 0.8),
                             feature_set=args.feature_set)
    _out(f"樣本 {rep['n']} 部：" + "、".join(f"{k} {v}" for k, v in rep["counts"].items()))
    _out(f"交叉驗證正確率：合併 {rep['accuracy']:.0%}（Qwen3-VL 單獨 {rep['qwen_only']:.0%}）")
    _out(f"信心 ≥ {opts.get('category_sure_prob', 0.8):.0%} 的有 {rep['sure_share']:.0%}，其中正確 {rep['accuracy_when_sure']:.0%}")
    _out("各類別找回率：" + "、".join(f"{k} {v:.0%}" for k, v in rep["per_class_recall"].items()))
    _out(f"已存到 {CATEGORY_CLASSIFIER}；設定檔 ai.category_model 設成 combined 就會使用")
    if not args.no_fallback and "frames_v3" not in args.feature_set:
        _out("訓練第二段（信心不足時加跑備援版）…")
        rep2 = evaluate_and_train(cfg, Analyzer(cfg, opts), progress=_out,
                                  sure_prob=opts.get("category_sure_prob", 0.8),
                                  feature_set=FALLBACK_FEATURE_SET + ("+audio_v1" if "audio" in args.feature_set else ""),
                                  out_path=FALLBACK_CLASSIFIER)
        _out(f"第二段交叉驗證正確率 {rep2['accuracy']:.0%}，已存到 {FALLBACK_CLASSIFIER}")
    return 0


def cmd_unknown(args, cfg: Config) -> int:
    from vsort.style_author import export_unknown
    out, n = export_unknown(cfg)
    _out(f"不知道作者的影片有 {n} 部，清單（附畫風相近的作者，僅供參考）：{out}")
    return 0


def cmd_authors(args, cfg: Config) -> int:
    from vsort.authors import accept_pending, update_authors
    if args.accept:
        acc, rej = accept_pending(cfg)
        _out(f"已接受 {acc} 筆、拒絕 {rej} 筆，別名表已更新：{cfg.dictionary_dir / 'authors.csv'}")
        return 0
    added, queued = update_authors(cfg)
    _out(f"新增 {added} 位作者到別名表，{queued} 筆疑似別名等你確認。")
    if queued:
        _out(f"請打開 {cfg.dictionary_dir / 'authors_pending.csv'}，在第一欄填 Y 或 N，存檔後執行 vsort authors --accept")
    return 0


def cmd_inspect(args, cfg: Config) -> int:
    """只看解析結果，不搬檔案；可以拿任何資料夾的影片來檢查規則。"""
    import csv
    from vsort.naming import build_final_name
    from vsort.parser import TYPE_LABEL
    from vsort.process import info_from_parsed
    from vsort.scanner import scan

    parser = make_parser(cfg)
    rows = []
    for raw in args.paths:
        base = Path(raw)
        res = scan([base], cfg)
        for sel in res.files:
            root = sel.root.parent if sel.root else sel.path.parent
            folders = list(sel.path.parent.relative_to(root).parts) if sel.root else []
            p = parser.parse(sel.path.name, folders)
            name = build_final_name(info_from_parsed(p), sel.path.suffix)
            rows.append((str(sel.path), TYPE_LABEL[p.name_type], p.site or p.site_guess or "",
                         p.source_id or "", name, p.category_hint or ""))
    for path, typ, site, sid, name, hint in rows[:args.limit]:
        _out(f"{path}\n   [{typ}{' ' + site if site else ''}{' ' + sid if sid else ''}] → {name}"
             + (f"   類別提示：{hint}" if hint else ""))
    if len(rows) > args.limit:
        _out(f"…共 {len(rows)} 部，其餘請看 CSV")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["原路徑", "檔名類型", "網站", "ID", "解析後檔名", "類別提示"])
            w.writerows(rows)
        _out(f"已輸出：{args.csv}")
    return 0


def cmd_apply(args, cfg: Config) -> int:
    plan = load_plan(cfg.layout, args.plan_id)
    if plan.applied_run:
        _out(f"計畫 {plan.plan_id} 已經執行過（紀錄 {plan.applied_run}），請重新試跑產生新計畫。")
        return 2
    _out(f"執行試跑計畫 {plan.plan_id}（{plan.kind}，{len(plan.ops)} 筆）")
    if plan.kind == "ingest" and plan.meta.get("batch"):
        return _run_plan(plan, cfg, batch=plan.meta["batch"])
    if plan.kind == "process":
        code = _run_plan(plan, cfg)
        _after_process(cfg)
        return code
    if plan.kind in ("analyze", "review", "rematch"):
        code = _run_plan(plan, cfg)
        _refresh_review(cfg)
        return code
    return _run_plan(plan, cfg, **({"undo_of": plan.meta["undo_of"],
                                    "partial": bool(plan.skipped)} if plan.kind == "undo" else {}))


def cmd_rematch(args, cfg: Config) -> int:
    """作者未知 的影片和現有作者比畫風指紋，夠像的送 05 並預填作者（預設試跑）。"""
    from vsort.rematch import plan_rematch
    from vsort.review import export
    cfg.layout.ensure()
    _out("比對 作者未知 的影片與已知作者的畫風（第一次可能要載入模型）…")
    plan, n = plan_rematch(cfg, min_similarity=args.min_similarity, min_margin=args.min_margin,
                           progress=lambda name, s: _out(f"  {name[:50]} → " + "、".join(f"{a} {v:.2f}" for a, v in s)))
    save_plan(plan, cfg.layout)
    _out(f"\n{'實際執行' if args.apply else '試跑（不會動任何檔案）'}：檢查 {n} 部，{len(plan.ops)} 部找到相似作者 → 05 人工確認")
    _show_plan(plan, cfg)
    if args.apply:
        code = _run_plan(plan, cfg)
        export(cfg)
        return code
    if plan.ops:
        _out(f"確認沒問題後執行：vsort apply {plan.plan_id}")
    return 0


def cmd_batches(args, cfg: Config) -> int:
    from vsort.batches import list_batches
    bs = list_batches(cfg)
    if not bs:
        _out("還沒有任何批次（從首頁或 vsort ingest 開始的每一批都會有編號）。")
    for b in bs:
        now = "、".join(f"{k} {v} 部" for k, v in sorted(b["now"].items())) or "（都已還原或不在工作區）"
        _out(f"  {b['id']}  {b['created']}  {'複製' if b['mode'] == 'copy' else '搬移'} {b['videos']} 部"
             f"{'（已全部還原）' if b['undone'] else ''}")
        _out(f"      來源：{'；'.join(b['sources'])}")
        _out(f"      現在：{now}；紀錄 {len(b['runs'])} 份")
    _out("整批還原：vsort undo --batch 批次編號 [--apply]")
    return 0


def cmd_undo(args, cfg: Config) -> int:
    if args.batch:
        from vsort.batches import plan_undo_batch, undo_batch
        plans = plan_undo_batch(cfg, args.batch)
        if not plans:
            _out(f"找不到批次 {args.batch} 還沒還原的紀錄。")
            return 2
        _out(f"{'實際還原' if args.apply else '試跑還原（不會動任何檔案）'}整批 {args.batch}：{len(plans)} 份紀錄")
        for run, plan, partial in plans:
            _out(f"\n紀錄 {run.run_id}（{ACTION_LABEL.get(run.kind, run.kind)}）：{len(plan.ops)} 筆"
                 + ("，只還原屬於這批的部分" if partial else ""))
            _show_plan(plan, cfg, limit=10)
        if args.apply:
            if not _guard_incomplete(cfg):
                return 2
            runs = undo_batch(cfg, args.batch, _progress)
            fails = sum(len(r.failed) for r in runs)
            _out(f"\n整批還原完成：{len(runs)} 份紀錄，失敗 {fails} 筆。")
            return 1 if fails else 0
        _out(f"\n確認後執行：vsort undo --batch {args.batch} --apply")
        return 0
    target = pick_undo_target(cfg.layout, args.run_id)
    if target.undone_by:
        _out(f"注意：{target.run_id} 已被 {target.undone_by.get('run')} 還原過。")
    plan = build_undo_plan(target)
    jpath, cpath = save_plan(plan, cfg.layout)
    _out(f"{'實際還原' if args.apply else '試跑還原（不會動任何檔案）'}：{target.run_id}，{len(plan.ops)} 筆")
    _show_plan(plan, cfg)
    _out(f"\n完整清單：{cpath}")
    if args.apply:
        return _run_plan(plan, cfg, undo_of=target.run_id, partial=bool(plan.skipped))
    _out(f"確認後執行：vsort undo {target.run_id} --apply")
    return 0


def cmd_resume(args, cfg: Config) -> int:
    pending = incomplete_runs(cfg.layout)
    retry = []
    if args.run_id:
        retry = [Run(cfg.layout.logs / f"{args.run_id}.jsonl")]
    elif not pending:
        failed = [r for r in list_runs(cfg.layout) if r.failed]
        if not failed:
            _out("沒有中斷或失敗待重試的紀錄。")
            return 0
        retry = failed[-1:]
    code = 0
    for run in pending or retry:
        _out(f"續跑 {run.run_id}（已完成 {len(run.done)}/{len(run.ops)}）")
        resume(run, _progress)
        _out(f"完成 {len(run.done)}/{len(run.ops)}，失敗 {len(run.failed)}。")
        code |= 1 if run.failed else 0
    return code


def cmd_runs(args, cfg: Config) -> int:
    runs = list_runs(cfg.layout)
    if not runs:
        _out("還沒有任何執行紀錄。")
    for r in runs[-args.limit:]:
        state = "已完成" if r.complete else "中斷"
        if r.failed:
            state += f"，{len(r.failed)} 筆失敗"
        extra = f"，已被 {r.undone_by['run']} 還原" if r.undone_by else ""
        if r.header.get("undo_of"):
            extra += f"，還原自 {r.header['undo_of']}"
        _out(f"  {r.run_id}  {r.created}  {len(r.done)}/{len(r.ops)} 筆  {state}{extra}")
    return 0


def cmd_status(args, cfg: Config) -> int:
    from vsort.process import _videos
    _out(f"工作區：{cfg.layout.workspace}")
    for label, path in cfg.layout.stage_dirs():
        _out(f"  {label}：{len(_videos(path, cfg))} 部")
    pending = incomplete_runs(cfg.layout)
    if pending:
        _out(f"  有 {len(pending)} 份中斷的紀錄，請執行 vsort resume")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vsort", description="影片自動分類系統")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", help="設定檔路徑（預設為專案資料夾裡的 config.yaml）")
    p.add_argument("--library", help="這次改用的成品收藏庫位置，例如 D:\\Library")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="建立工作區資料夾")

    s = sub.add_parser("ingest", help="把挑選的影片放進 01_Inbox（預設試跑）")
    s.add_argument("paths", nargs="*", help="影片檔或資料夾，可多個")
    s.add_argument("--pick", action="store_true", help="開視窗挑選影片檔")
    s.add_argument("--pick-folder", action="store_true", help="開視窗挑選資料夾")
    s.add_argument("--copy", action="store_true", help="用複製代替搬移，原檔保留")
    s.add_argument("--no-recursive", action="store_true", help="資料夾只取第一層")
    s.add_argument("--keep-duplicates", action="store_true", help="和工作區或收藏庫重複的也放進來")
    s.add_argument("--apply", action="store_true", help="真的執行")

    s = sub.add_parser("process", help="解析 01_Inbox 的檔名、查詢來源網站並分流到 03（預設試跑）")
    s.add_argument("--apply", action="store_true", help="真的執行")
    s.add_argument("--offline", action="store_true", help="不連網，只用檔名解析")

    s = sub.add_parser("analyze", help="AI 分析 03 的影片並判定：已確定進 04，可能誤判進 05（預設試跑）")
    s.add_argument("--apply", action="store_true", help="真的執行")
    s.add_argument("--library", action="store_true", help="也重新分析 04 裡類別或標籤不全的影片")

    s = sub.add_parser("review", help="產生 05 人工確認清單；--import 依你填的內容歸檔")
    s.add_argument("--import", dest="do_import", action="store_true")
    s.add_argument("--web", action="store_true", help="開啟人工確認網頁（看預覽圖、改欄位、按確認歸檔）")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--apply", action="store_true")

    s = sub.add_parser("web", help="開網頁首頁：在瀏覽器挑影片或資料夾、一鍵分類、人工確認")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true", help="不要自動開瀏覽器")

    s = sub.add_parser("dupes", help="列出重複的影片（預設檢查工作區與收藏庫，不刪除）")
    s.add_argument("paths", nargs="*")
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("label", help="開標註網頁，替抽樣的影片勾動畫類型（給自訓分類器用）")
    s.add_argument("paths", nargs="+", help="要抽樣的資料夾")
    s.add_argument("--n", type=int, default=150, help="這次抽幾部（預設 150）")
    s.add_argument("--per-folder", type=int, default=8, help="每個作者資料夾最多抽幾部")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--offline", action="store_true", help="不查 iwara 標籤")
    s.add_argument("--port", type=int, default=8767)
    s.add_argument("--no-browser", action="store_true", help="不要自動開瀏覽器")
    s.add_argument("--uncertain", action="store_true", help="挑目前分類器最沒把握的影片（第二輪以後用）")
    s.add_argument("--include-labeled", help="也列出已標過這些類型的影片以便改標，例如 3D")
    s.add_argument("--category-uncertain", action="store_true", help="挑最可能是 Mix 或類別最沒把握的影片")

    s = sub.add_parser("train-style", help="比較 CLIP、SigLIP、小分類器判斷動畫類型的正確率，並訓練最好的")
    s.add_argument("--features", default="clip,siglip,pixai_style")
    s.add_argument("--no-train", action="store_true", help="只比較，不存模型")

    sub.add_parser("unknown", help="列出不知道作者的影片，附畫風相近的作者（僅供參考）")
    s = sub.add_parser("train-category", help="用你標的類別訓練類別分類器（畫面標籤＋Qwen3-VL）")
    s.add_argument("--feature-set", default="tags+frames_v1+video_v1+audio_v1",
                   choices=["tags+frames_v1+video_v1", "tags+frames_v1+video_v1+frames_v3",
                            "tags+frames_v1+video_v1+audio_v1", "tags+frames_v1+video_v1+frames_v3+audio_v1"],
                   help="要用哪些判斷當特徵；audio_v1＝聲音特徵（每部約 1.5 秒）；frames_v3＝備援版（每部多約 10 秒）")
    s.add_argument("--no-fallback", action="store_true", help="不訓練「信心不足時加跑備援版」的第二段")

    s = sub.add_parser("authors", help="更新作者別名表；--accept 套用你在待確認清單填的 Y/N")
    s.add_argument("--accept", action="store_true")

    s = sub.add_parser("inspect", help="只顯示檔名解析結果，不搬任何檔案")
    s.add_argument("paths", nargs="+", help="影片檔或資料夾")
    s.add_argument("--csv", help="另存成 CSV")
    s.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("apply", help="執行先前試跑產生的計畫")
    s.add_argument("plan_id", nargs="?", help="計畫編號，預設最新一份")

    s = sub.add_parser("undo", help="還原一次執行或整批（預設試跑）")
    s.add_argument("run_id", nargs="?", help="紀錄編號，預設最近一次尚未還原的")
    s.add_argument("--batch", help="整批還原：批次編號（vsort batches 可查），含之後的分流、歸檔、人工確認")
    s.add_argument("--apply", action="store_true", help="真的還原")

    sub.add_parser("batches", help="列出歷史批次：來源、部數、現在各在哪")

    s = sub.add_parser("rematch", help="作者未知 的影片重新用畫風指紋比對已知作者，夠像的送 05 預填作者（預設試跑）")
    s.add_argument("--apply", action="store_true", help="真的執行")
    s.add_argument("--min-similarity", type=float, default=0.5, help="相似度門檻（預設 0.5）")
    s.add_argument("--min-margin", type=float, default=0.1, help="要領先第二名多少（預設 0.1）")

    s = sub.add_parser("resume", help="續跑中斷的紀錄，或重試失敗的項目")
    s.add_argument("run_id", nargs="?")

    s = sub.add_parser("runs", help="列出執行紀錄")
    s.add_argument("--limit", type=int, default=20)

    sub.add_parser("status", help="各資料夾的影片數量")
    return p


COMMANDS = {"init": cmd_init, "web": cmd_web, "batches": cmd_batches, "rematch": cmd_rematch, "ingest": cmd_ingest, "process": cmd_process, "authors": cmd_authors, "label": cmd_label, "train-style": cmd_train_style,
            "train-category": cmd_train_category, "unknown": cmd_unknown, "dupes": cmd_dupes, "analyze": cmd_analyze, "review": cmd_review, "inspect": cmd_inspect, "apply": cmd_apply, "undo": cmd_undo,
            "resume": cmd_resume, "runs": cmd_runs, "status": cmd_status}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.library)
        return COMMANDS[args.cmd](args, cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        _out(f"錯誤：{exc}")
        return 2
