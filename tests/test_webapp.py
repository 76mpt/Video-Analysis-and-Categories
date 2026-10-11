"""網頁首頁：瀏覽資料夾、挑影片、一鍵跑完整流程（用假的 AI）。"""

import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from conftest import make_video
from test_analyze import FakeStyle, FakeTagger, fake_extract, frames
from vsort.analyze import DEFAULTS, Analyzer
from vsort.pipeline import Job
from vsort.webapp import Handler, browse


def _server(cfg, job):
    handler = type("H", (Handler,), {"cfg": cfg, "job": job})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    return json.load(urllib.request.urlopen(url))


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_browse_lists_dirs_and_videos(cfg, tmp_path):
    src = tmp_path / "src"
    make_video(src / "The Studio" / "Wiggle.mp4")
    (src / "The Studio" / "notes.txt").write_text("x")
    make_video(src / "other" / "b.mkv")
    d = browse(cfg, str(src))
    assert [x["name"] for x in d["dirs"]] == ["other", "The Studio"] and d["files"] == []
    d = browse(cfg, str(src / "The Studio"))
    assert [x["name"] for x in d["files"]] == ["Wiggle.mp4"] and d["parent"] == str(src)
    assert browse(cfg, "")["dirs"]  # 磁碟機清單
    # 工作區不會列出
    names = [x["name"] for x in browse(cfg, str(cfg.layout.workspace.parent))["dirs"]]
    assert cfg.layout.workspace.name not in names


def test_home_pipeline_end_to_end(cfg, tmp_path):
    src = tmp_path / "src"
    a = make_video(src / "The Studio" / "Wiggle.mp4")
    b = make_video(src / "untitled.mp4", size=3000)
    tagger = FakeTagger({"Wiggle": frames(16, dancing=0.9, hatsune_miku=0.97),
                         "untitled": frames(16, scene_tag=0.9, **{"shiranui_(kancolle)": 0.7})})
    analyzer = Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())
    job = Job()
    httpd, base = _server(cfg, job)
    try:
        assert "影片分類" in urllib.request.urlopen(base + "/").read().decode()
        assert "人工確認" in urllib.request.urlopen(base + "/review").read().decode()
        code, res = _post(base + "/api/start", {"paths": [], "copy": True})
        assert code == 400
        job._analyzer = analyzer  # 測試不載真模型
        code, res = _post(base + "/api/start", {"paths": [str(src / "The Studio"), str(b)], "copy": True})
        assert code == 200, res
        code, res = _post(base + "/api/start", {"paths": [str(b)], "copy": True})
        assert code == 409  # 一次只跑一件

        def wait_state(*states):
            for _ in range(200):
                st = _get(base + "/api/status")["job"]
                if st["state"] in states:
                    return st
                time.sleep(0.05)
            raise AssertionError(st)

        # 第一個確認點：放進 01 的試跑清單
        st = wait_state("waiting", "error")
        assert st["state"] == "waiting" and st["pending"]["kind"] == "ingest", st
        assert len(st["pending"]["rows"]) == 2 and st["pending"]["action"] == "複製"
        assert not list(cfg.layout.inbox.rglob("*.mp4"))  # 還沒動檔案
        code, res = _post(base + "/api/answer", {"accept": True})
        assert code == 200
        # 第二個確認點：AI 判定的試跑清單
        st = wait_state("waiting", "error", "done")
        assert st["state"] == "waiting" and st["pending"]["kind"] == "analyze", st
        rows = st["pending"]["rows"]
        assert st["pending"]["library"] == 1 and st["pending"]["review"] == 1
        wiggle = next(r for r in rows if r["name"].startswith("Wiggle"))
        assert wiggle["to"] == "library" and wiggle["dst_name"] == "Wiggle [The Studio][VOCALOID][Hatsune Miku][MMD][Dance].mp4"
        # 把已確定的那部改送人工確認
        code, res = _post(base + "/api/answer", {"accept": True, "to_review": [wiggle["src"]]})
        assert code == 200
        job.wait(30)
        st = _get(base + "/api/status")
        assert st["job"]["state"] == "done", st["job"]
        assert st["job"]["counts"] == {"library": 0, "review": 2, "rematched": 0}
        assert a.exists() and b.exists()  # 複製模式，原檔都在
        assert not list(cfg.layout.library.rglob("*.mp4"))
        items = _get(base + "/api/items")["items"]
        assert len(items) == 2 and any("勾選了人工確認" in r for it in items for r in it["reasons"])
    finally:
        httpd.shutdown()


def test_pick_dialog_route(cfg, tmp_path, monkeypatch):
    from vsort import webapp
    v = make_video(tmp_path / "src" / "a.mp4")
    monkeypatch.setattr(webapp, "pick_in_subprocess", lambda kind: [str(v), str(tmp_path / "src"), str(tmp_path / "nope.txt")])
    httpd, base = _server(cfg, Job())
    try:
        code, res = _post(base + "/api/pick", {"kind": "nope"})
        assert code == 400
        code, res = _post(base + "/api/pick", {"kind": "files"})
        assert code == 200 and [x["kind"] for x in res["items"]] == ["file", "dir"]
    finally:
        httpd.shutdown()


def test_home_pipeline_cancel(cfg, tmp_path):
    src = tmp_path / "src"
    a = make_video(src / "The Studio" / "Wiggle.mp4")
    job = Job()
    httpd, base = _server(cfg, job)
    try:
        code, res = _post(base + "/api/start", {"paths": [str(a)], "copy": False})
        assert code == 200, res
        for _ in range(200):
            if _get(base + "/api/status")["job"]["state"] == "waiting":
                break
            time.sleep(0.05)
        code, res = _post(base + "/api/answer", {"accept": False})
        assert code == 200
        job.wait(10)
        st = _get(base + "/api/status")["job"]
        assert st["state"] == "cancelled" and a.exists() and not list(cfg.layout.inbox.rglob("*.mp4"))
        code, res = _post(base + "/api/answer", {"accept": True})
        assert code == 409
    finally:
        httpd.shutdown()


def test_batch_undo_restores_everything(cfg, tmp_path):
    """整批還原：搬移的搬回原位置、複製的刪掉複本，含人工確認過的；資訊檔記 original_path 與 batch。"""
    from vsort import sidecar as sc
    from vsort.batches import list_batches
    src = tmp_path / "src"
    a = make_video(src / "The Studio" / "Wiggle.mp4")
    b = make_video(src / "untitled.mp4", size=3000)
    tagger = FakeTagger({"Wiggle": frames(16, dancing=0.9, hatsune_miku=0.97),
                         "untitled": frames(16, scene_tag=0.9, **{"shiranui_(kancolle)": 0.7})})
    analyzer = Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())
    job = Job()
    job._analyzer = analyzer
    httpd, base = _server(cfg, job)

    def wait_state(*states):
        for _ in range(300):
            st = _get(base + "/api/status")["job"]
            if st["state"] in states:
                return st
            time.sleep(0.05)
        raise AssertionError(st)

    try:
        code, res = _post(base + "/api/start", {"paths": [str(src / "The Studio"), str(b)], "copy": False})  # 搬移
        assert code == 200, res
        wait_state("waiting"); _post(base + "/api/answer", {"accept": True})
        wait_state("waiting"); _post(base + "/api/answer", {"accept": True})
        job.wait(30)
        st = _get(base + "/api/status")["job"]
        assert st["state"] == "done" and st["batch"].startswith("batch-"), st
        assert not a.exists() and not b.exists()  # 搬移模式，原檔已進工作區
        lib = list(cfg.layout.library.rglob("*.mp4"))
        assert len(lib) == 1 and sc.read(lib[0])["original_path"] == str(a) and sc.read(lib[0])["batch"] == st["batch"]
        # 人工確認那一部
        items = _get(base + "/api/items")["items"]
        code, res = _post(base + "/api/confirm", {"name": items[0]["name"], "info": {
            "title": "untitled", "author": "Mochi", "works": "艦隊收藏", "characters": "不知火", "anim_type": "3D", "category": "Scene"}})
        assert code == 200, res
        assert len(list(cfg.layout.library.rglob("*.mp4"))) == 2
        bs = _get(base + "/api/batches")["batches"]
        assert len(bs) == 1 and bs[0]["id"] == st["batch"] and bs[0]["videos"] == 2 and bs[0]["now"] == {"04": 2}
        assert [r["kind"] for r in bs[0]["runs"]] == ["ingest", "process", "analyze", "review"]
        # 試跑還原 → 真的還原
        code, res = _post(base + "/api/undo_batch", {"batch": st["batch"], "apply": False})
        assert code == 200 and [p["kind"] for p in res["plans"]] == ["review", "analyze", "process", "ingest"], res
        code, res = _post(base + "/api/undo_batch", {"batch": st["batch"], "apply": True})
        assert code == 200, res
        job.wait(30)
        st2 = _get(base + "/api/status")["job"]
        assert st2["state"] == "done" and st2["counts"]["restored"] == 7, st2  # 人工確認 1＋歸檔 2＋分流 2＋放進 01 的 2
        assert a.exists() and b.exists()
        assert not list(cfg.layout.workspace.rglob("*.mp4")) and not list(cfg.layout.library.rglob("*.mp4"))
        assert not list(src.rglob("*.vsort.json"))  # 原位置不留資訊檔
        bs = _get(base + "/api/batches")["batches"]
        assert bs[0]["undone"] and bs[0]["now"] == {}
        code, res = _post(base + "/api/undo_batch", {"batch": st["batch"], "apply": False})
        assert code == 404
    finally:
        httpd.shutdown()


def test_root_dir_layout_and_change(cfg, tmp_path):
    """root_dir：sorting 底下是工作區、作者資料夾在根目錄；收藏庫掃描不會把 sorting 裡的影片算進去；網頁可改位置。"""
    import yaml
    from vsort.config import load_config
    from vsort.process import _videos
    root = tmp_path / "VIDEO"
    root.mkdir()
    conf = dict(yaml.safe_load(cfg.path.read_text(encoding="utf-8")))
    conf["root_dir"] = str(root)
    cfg.path.write_text(yaml.safe_dump(conf, allow_unicode=True), encoding="utf-8")
    c2 = load_config(cfg.path)
    assert c2.layout.workspace == (root / "sorting").resolve() and c2.layout.library == root.resolve()
    assert c2.layout.inbox == (root / "sorting" / "01_Inbox").resolve()
    c2.layout.ensure()
    lib_v = make_video(root / "Mochi" / "a [Mochi][-][-][3D][Scene].mp4")
    make_video(c2.layout.inbox / "b.mp4")
    assert _videos(c2.layout.library, c2) == [lib_v]
    # 網頁改位置：工作區有影片時不給改；清空後可以改，設定檔只改 root_dir 這一行
    httpd, base = _server(c2, Job())
    try:
        st = _get(base + "/api/status")
        assert st["paths"]["root"] == str(root.resolve()) and dict(st["stages"])["04 收藏庫"] == 1
        other = tmp_path / "OTHER"
        other.mkdir()
        code, res = _post(base + "/api/set_root", {"path": str(other)})
        assert code == 409
        (c2.layout.inbox / "b.mp4").unlink()
        code, res = _post(base + "/api/set_root", {"path": str(other)})
        assert code == 200, res
        assert _get(base + "/api/status")["paths"]["library"] == str(other.resolve())
        assert f"root_dir: '{other}'" in cfg.path.read_text(encoding="utf-8")
        code, res = _post(base + "/api/set_root", {"path": str(tmp_path / "nope")})
        assert code == 400
    finally:
        httpd.shutdown()
