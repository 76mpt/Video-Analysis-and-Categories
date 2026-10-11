import json
from pathlib import Path

import pytest

from conftest import make_video
from vsort import fileops, journal
from vsort.cli import main
from vsort.ingest import plan_ingest
from vsort.journal import Run, build_undo_plan, execute, incomplete_runs, list_runs, resume


def test_scan_filters(cfg, tmp_path):
    src = tmp_path / "src"
    make_video(src / "ok.mp4")
    make_video(src / "sub" / "deep.mkv")
    make_video(src / "dl.mp4.part")
    make_video(src / "busy.mp4")
    make_video(src / "busy.mp4.crdownload")
    make_video(src / "new.mp4", age=1)
    make_video(src / "cover.jpg")
    (src / "empty.mp4").write_bytes(b"")
    plan = plan_ingest([src], cfg)
    names = sorted(Path(o.src).name for o in plan.ops)
    assert names == ["deep.mkv", "ok.mp4"]
    reasons = dict((Path(p).name, r) for p, r in plan.skipped)
    assert "暫存" in reasons["dl.mp4.part"]
    assert "下載" in reasons["busy.mp4"]
    assert "寫入" in reasons["new.mp4"]
    assert "0" in reasons["empty.mp4"]
    assert plan.meta["non_video_count"] == 1


def test_keeps_folder_structure(cfg, tmp_path):
    src = tmp_path / "作者A"
    make_video(src / "sub" / "v.mp4")
    plan = plan_ingest([src], cfg)
    assert Path(plan.ops[0].dst) == cfg.layout.inbox / "作者A" / "sub" / "v.mp4"


def test_skip_workspace(cfg):
    make_video(cfg.layout.inbox / "already.mp4")
    plan = plan_ingest([cfg.layout.inbox / "already.mp4"], cfg)
    assert not plan.ops and "工作區" in plan.skipped[0][1]


def test_dry_run_touches_nothing(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    rc = main(["--config", str(cfg.path), "ingest", str(v)])
    assert rc == 0 and v.exists()
    assert not list(cfg.layout.inbox.rglob("*.mp4"))
    assert list(cfg.layout.reports.glob("plan-*.csv"))
    assert not list_runs(cfg.layout)


def test_apply_then_undo(cfg, tmp_path):
    a = make_video(tmp_path / "src" / "a:?.mp4".replace(":?", "x"))
    b = make_video(tmp_path / "other" / "a.mp4")
    assert main(["--config", str(cfg.path), "ingest", str(a), str(b), "--apply"]) == 0
    assert not a.exists() and not b.exists()
    inbox = sorted(p.name for p in cfg.layout.inbox.rglob("*.mp4"))
    assert len(inbox) == 2
    # 還原（先試跑，再實際）
    assert main(["--config", str(cfg.path), "undo"]) == 0
    assert not a.exists()
    assert main(["--config", str(cfg.path), "undo", "--apply"]) == 0
    assert a.exists() and b.exists()
    assert not list(cfg.layout.inbox.rglob("*.mp4"))
    orig = [r for r in list_runs(cfg.layout) if r.kind == "ingest"][0]
    assert orig.undone_by is not None
    # 沒有其他可還原的
    assert main(["--config", str(cfg.path), "undo"]) == 2


def test_apply_saved_plan(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    main(["--config", str(cfg.path), "ingest", str(v)])
    assert main(["--config", str(cfg.path), "apply"]) == 0
    assert not v.exists()
    assert main(["--config", str(cfg.path), "apply"]) == 2  # 同一份計畫不能執行兩次


def test_never_overwrites(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    plan = plan_ingest([v], cfg)
    make_video(Path(plan.ops[0].dst), size=10)  # 計畫後有人放了同名檔
    run = Run.create(cfg.layout, "ingest", plan.ops)
    execute(run)
    assert run.failed and v.exists()
    assert Path(plan.ops[0].dst).stat().st_size == 10


def test_resume_after_crash(cfg, tmp_path, monkeypatch):
    vids = [make_video(tmp_path / "src" / f"v{i}.mp4") for i in range(4)]
    plan = plan_ingest(vids, cfg)
    run = Run.create(cfg.layout, "ingest", plan.ops)

    calls = {"n": 0}
    real = journal._perform

    def crashing(op):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # 模擬執行到一半被關掉
        real(op)

    monkeypatch.setattr(journal, "_perform", crashing)
    with pytest.raises(KeyboardInterrupt):
        execute(run)
    monkeypatch.setattr(journal, "_perform", real)

    pending = incomplete_runs(cfg.layout)
    assert len(pending) == 1
    reloaded = pending[0]
    assert len(reloaded.done) == 2 and 2 in reloaded.begun
    # 有中斷紀錄時，新的執行會被擋下
    other = make_video(tmp_path / "src2" / "x.mp4")
    assert main(["--config", str(cfg.path), "ingest", str(other), "--apply"]) == 2
    resume(reloaded)
    assert reloaded.complete and len(reloaded.done) == 4
    assert all(not v.exists() for v in vids)


def test_cross_volume_move_and_crash_between_copy_and_delete(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(fileops, "same_volume", lambda a, b: False)
    v = make_video(tmp_path / "src" / "big.mp4", size=3 * 1024 * 1024)
    plan = plan_ingest([v], cfg)
    run = Run.create(cfg.layout, "ingest", plan.ops)
    dst = Path(plan.ops[0].dst)

    # 模擬：複本已核對改名，但刪原檔前當機
    run.write({"type": "begin", "i": 0})
    fileops.copy_verified(v, dst)
    reloaded = Run(run.path)
    execute(reloaded)
    assert reloaded.complete and not reloaded.failed
    assert dst.exists() and not v.exists()


def test_cross_volume_partial_copy_is_redone(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(fileops, "same_volume", lambda a, b: False)
    v = make_video(tmp_path / "src" / "big.mp4", size=3 * 1024 * 1024)
    plan = plan_ingest([v], cfg)
    run = Run.create(cfg.layout, "ingest", plan.ops)
    dst = Path(plan.ops[0].dst)
    run.write({"type": "begin", "i": 0})
    dst.parent.mkdir(parents=True, exist_ok=True)
    fileops.part_path(dst).write_bytes(b"half")  # 複製到一半
    reloaded = Run(run.path)
    execute(reloaded)
    assert dst.exists() and not v.exists() and not fileops.part_path(dst).exists()
    assert dst.stat().st_size == 3 * 1024 * 1024


def test_copy_mode_and_undo(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    assert main(["--config", str(cfg.path), "ingest", str(v), "--copy", "--apply"]) == 0
    assert v.exists() and list(cfg.layout.inbox.rglob("a.mp4"))
    assert main(["--config", str(cfg.path), "undo", "--apply"]) == 0
    assert v.exists() and not list(cfg.layout.inbox.rglob("a.mp4"))


def test_undo_skips_files_moved_later(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    main(["--config", str(cfg.path), "ingest", str(v), "--apply"])
    moved = list(cfg.layout.inbox.rglob("a.mp4"))[0]
    moved.rename(cfg.layout.review / "a.mp4")
    run = [r for r in list_runs(cfg.layout) if r.kind == "ingest"][0]
    plan = build_undo_plan(run)
    assert not plan.ops and plan.skipped


def test_truncated_last_line_is_ignored(cfg, tmp_path):
    v = make_video(tmp_path / "src" / "a.mp4")
    plan = plan_ingest([v], cfg)
    run = Run.create(cfg.layout, "ingest", plan.ops)
    with open(run.path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "beg')
    reloaded = Run(run.path)
    execute(reloaded)
    assert reloaded.complete and not v.exists()


def test_library_override(cfg, tmp_path):
    from vsort.config import load_config
    c = load_config(cfg.path, str(tmp_path / "lib"))
    assert c.layout.library == (tmp_path / "lib").resolve()


def test_duplicates_skipped(cfg, tmp_path):
    import shutil
    a = make_video(tmp_path / "src" / "a.mp4", size=5 * 1024 * 1024)
    b = tmp_path / "src2" / "a copy.mp4"
    b.parent.mkdir()
    shutil.copy2(a, b)
    lib_copy = cfg.layout.library / "X" / "old name.mp4"
    lib_copy.parent.mkdir(parents=True)
    other = make_video(tmp_path / "src3" / "c.mp4", size=5 * 1024 * 1024)  # 同大小不同內容
    # 兩份相同的一起挑：只放一份
    plan = plan_ingest([a, b, other], cfg)
    assert [Path(o.src).name for o in plan.ops] == ["a.mp4", "c.mp4"]
    assert "重複" in plan.skipped[0][1]
    # 收藏庫已有 → 略過
    shutil.copy2(a, lib_copy)
    plan2 = plan_ingest([a], cfg)
    assert not plan2.ops and "old name" in plan2.skipped[0][1]
    assert len(plan_ingest([a], cfg, skip_duplicates=False).ops) == 1


def test_dupes_command(cfg, tmp_path):
    import shutil
    a = make_video(cfg.layout.library / "A" / "x.mp4")
    shutil.copy2(a, cfg.layout.review / "y.mp4")
    assert main(["--config", str(cfg.path), "dupes"]) == 0
    assert list(cfg.layout.reports.glob("dupes-*.csv"))
    assert a.exists() and (cfg.layout.review / "y.mp4").exists()
