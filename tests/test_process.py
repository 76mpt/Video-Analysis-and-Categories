from pathlib import Path

from conftest import make_video
from vsort import sidecar as sc
from vsort.cli import main
from vsort.journal import list_runs
from vsort.process import plan_process


def test_process_routes_and_undo(cfg, tmp_path):
    src = tmp_path / "src"
    make_video(src / "Mochi" / "88aa854e_がんばる天津風.mp4")
    make_video(src / "未分類" / "untitled thing.mp4")
    make_video(src / "iwara" / "Iwara - Poison [Xy12Ab34Cd56Ef] [Source].mp4")
    c = ["--config", str(cfg.path)]
    assert main(c + ["ingest", str(src / "Mochi"), str(src / "未分類"), str(src / "iwara"), "--apply"]) == 0

    plan, counts = plan_process(cfg)
    assert counts == {"refine": 2, "identify": 1}
    # 試跑不動檔案
    assert main(c + ["process"]) == 0
    assert len(list(cfg.layout.inbox.rglob("*.mp4"))) == 3

    assert main(c + ["process", "--apply"]) == 0
    refine = sorted(p.name for p in cfg.layout.refine.glob("*.mp4"))
    assert refine == ["Poison [-][-][-][-][-].mp4", "がんばる天津風 [Mochi][艦隊收藏][Amatsukaze][-][-].mp4"]
    ident = list(cfg.layout.identify.glob("*.mp4"))
    assert [p.name for p in ident] == ["untitled thing [-][-][-][-][-].mp4"]
    data = sc.read(cfg.layout.refine / refine[0])
    assert data["parsed"]["source_id"] == "Xy12Ab34Cd56Ef"
    assert data["original_name"] == "Iwara - Poison [Xy12Ab34Cd56Ef] [Source].mp4"
    assert data["stage"] == "refine"

    # 再跑一次不會有任何動作（03 不在掃描範圍）
    plan2, _ = plan_process(cfg)
    assert not plan2.ops

    # 還原：影片回到 Inbox，資訊檔退回放進 01 時的內容（只剩原始路徑與批次編號）
    assert main(c + ["undo", "--apply"]) == 0
    back = list(cfg.layout.inbox.rglob("*.mp4"))
    assert len(back) == 3
    for v in back:
        d = sc.read(v)
        assert "parsed" not in d and d["original_path"] and d["batch"].startswith("batch-")
    assert not list(cfg.layout.refine.iterdir())
    # 再還原 ingest：原檔回到原位置，資訊檔不留下
    assert main(c + ["undo", "--apply"]) == 0
    assert (src / "Mochi" / "88aa854e_がんばる天津風.mp4").exists()
    assert not list(src.rglob("*.vsort.json"))


def test_inspect_does_not_move(cfg, tmp_path):
    v = make_video(tmp_path / "Mochi" / "88aa854e_天津風.mp4")
    out = tmp_path / "r.csv"
    assert main(["--config", str(cfg.path), "inspect", str(tmp_path / "Mochi"), "--csv", str(out)]) == 0
    assert v.exists() and out.exists()
    assert "Mochi" in out.read_text(encoding="utf-8-sig")
    assert not list_runs(cfg.layout)
