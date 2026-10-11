"""動畫類型標註與自訓分類器：用假的特徵（不需要模型）驗證抽樣、存檔、交叉驗證、訓練與推論。"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

from conftest import make_video
from vsort import style_train
from vsort.labeling import Handler, LabelSession, load_labels, prefill, sample_videos, save_labels
from vsort.process import make_parser


def test_sample_spreads_over_folders(cfg, tmp_path):
    for a in ("A", "B", "C"):
        for i in range(10):
            make_video(tmp_path / "lib" / a / f"{a}{i}.mp4")
    picked = sample_videos([tmp_path / "lib"], cfg, n=9, per_folder=3, seed=1, exclude=set())
    assert len(picked) == 9
    assert sorted({p.parent.name for p in picked}) == ["A", "B", "C"]
    assert all(sum(p.parent.name == a for p in picked) == 3 for a in "ABC")


def test_prefill_from_filename(cfg, tmp_path):
    v = make_video(tmp_path / "lib" / "X" / "【MMD】dance blender.mp4")
    labels, src = prefill(v, tmp_path / "lib", make_parser(cfg))
    assert labels == ["MMD", "3D"] and src == "檔名"


def test_label_session_saves_csv(cfg, tmp_path):
    v = make_video(tmp_path / "lib" / "X" / "a.mp4")
    queue = [{"key": "k1", "path": str(v), "name": v.name, "folder": "X", "prefill": ["MMD"], "source": "檔名"}]
    s = LabelSession(cfg, queue, frames=2)
    s.set_label("k1", ["3D", "MMD"], False)
    got = load_labels(cfg)
    assert got["k1"]["labels"] == ["MMD", "3D"] and not got["k1"]["skip"]
    text = (cfg.dictionary_dir / "style_labels.csv").read_text(encoding="utf-8-sig")
    assert "看不出來" in text and str(v) in text

    handler = type("H", (Handler,), {"session": s})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        items = json.load(urllib.request.urlopen(base + "/api/items"))["items"]
        assert items[0]["done"] and items[0]["labels"] == ["MMD", "3D"]
        req = urllib.request.Request(base + "/api/label", data=json.dumps({"key": "k1", "skip": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)
        assert load_labels(cfg)["k1"]["skip"]
    finally:
        httpd.shutdown()


class FakeEmb(style_train.Embedder):
    kind = "fake"

    def embed(self, images):
        # 用圖片顏色當特徵：紅＝MMD、綠＝3D、藍＝2D
        return np.stack([np.asarray(im.resize((1, 1))).reshape(-1) / 255.0 for im in images])


def _make_dataset(cfg, n=30):
    rng = np.random.default_rng(0)
    labels = {}
    for i in range(n):
        lab = [["MMD"], ["3D"], ["2D"], ["MMD", "3D"]][i % 4]
        color = (200 if "MMD" in lab else 30, 200 if "3D" in lab else 30, 200 if "2D" in lab else 30)
        key = f"key{i:03d}"
        fdir = cfg.layout.workspace / "_cache" / "frames" / key
        fdir.mkdir(parents=True)
        for j in range(2):
            c = tuple(int(x + rng.integers(-20, 20)) for x in color)
            Image.new("RGB", (8, 8), c).save(fdir / f"{j:02d}.jpg")
        (fdir / "frames.json").write_text(json.dumps({"count": 2, "times": [0, 1]}), encoding="utf-8")
        labels[key] = {"path": str(fdir / "missing.mp4"), "labels": lab, "skip": False, "source": "", "t": ""}
    save_labels(cfg, labels)


def test_cv_and_train_roundtrip(cfg, tmp_path, monkeypatch):
    _make_dataset(cfg)
    monkeypatch.setitem(style_train.EMBEDDERS, "fake", FakeEmb)
    monkeypatch.setattr(style_train, "CLASSIFIER_PATH", tmp_path / "style_classifier.json")
    rep = style_train.evaluate(cfg, frames=2, features=("fake",), progress=lambda *_: None)
    m = rep["methods"]["小分類器（fake 特徵）"]
    assert m["exact"] >= 0.9 and m["F1_MMD"] >= 0.9
    path = style_train.train(cfg, "fake", m["C"], frames=2)
    clf = style_train.TrainedStyleClassifier(cfg.layout.workspace / "_cache", path)
    both = clf.predict_video("new", [Image.new("RGB", (8, 8), (200, 200, 30))] * 2)
    assert both["MMD"] > 0.5 and both["3D"] > 0.5 and both["2D"] < 0.5


def test_too_few_samples(cfg):
    import pytest
    with pytest.raises(RuntimeError):
        style_train.evaluate(cfg, features=(), progress=lambda *_: None)


def test_pick_uncertain(cfg, tmp_path, monkeypatch):
    from vsort import labeling
    vids = [make_video(tmp_path / "lib" / "A" / f"v{i}.mp4", size=2048 + i) for i in range(4)]
    probs = {0: {"MMD": 0.99, "3D": 0.01, "2D": 0.01},   # 很有把握
             1: {"MMD": 0.55, "3D": 0.45, "2D": 0.01},   # 最沒把握
             2: {"MMD": 0.9, "3D": 0.3, "2D": 0.01},     # 有點像 3D（樣本少，優先）
             3: {"MMD": 0.98, "3D": 0.02, "2D": 0.0}}

    class FakeClf:
        def __init__(self, cache):
            pass

        def predict_video(self, key, imgs):
            return probs[imgs[0]]

    monkeypatch.setattr(style_train, "TrainedStyleClassifier", FakeClf)
    monkeypatch.setattr(style_train, "CLASSIFIER_PATH", tmp_path / "exists.json")
    (tmp_path / "exists.json").write_text("{}")
    monkeypatch.setattr(labeling, "extract_cached", lambda p, n, d: [(0, int(p.stem[1:]))])
    picked = labeling.pick_uncertain(cfg, vids, 2, progress=lambda *_: None)
    assert [p.name for p in picked] == ["v1.mp4", "v2.mp4"]


def test_pick_category_uncertain(cfg, tmp_path):
    from vsort import labeling
    from vsort.analyze import AIResult
    vids = [make_video(tmp_path / "lib" / "A" / f"v{i}.mp4", size=2048 + i) for i in range(4)]
    probs = {0: {"Scene": 0.95, "Mix": 0.03, "Dance": 0.02}, 1: {"Scene": 0.5, "Mix": 0.45, "Dance": 0.05},
             2: {"Dance": 0.9, "Mix": 0.05, "Scene": 0.05}, 3: {"Scene": 0.4, "Dance": 0.35, "Mix": 0.25}}

    class FakeAn:
        def run(self, p):
            return AIResult(category_probs=probs[int(p.stem[1:])], category="Scene")

    picked = labeling.pick_category_uncertain(cfg, vids, 2, progress=lambda *_: None, analyzer=FakeAn())
    assert [p.name for p in picked] == ["v1.mp4", "v3.mp4"]


def test_sample_skips_folders(cfg, tmp_path):
    from vsort.labeling import sample_videos
    cfg.data["skip_folders"] = ["LiveAction"]
    good = make_video(tmp_path / "lib" / "A" / "a.mp4")
    make_video(tmp_path / "lib" / "用" / "liveaction" / "real.mp4")
    picked = sample_videos([tmp_path / "lib"], cfg, 10, 8, 1, set())
    assert picked == [good]
