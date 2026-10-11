"""依畫風推薦作者：只當參考，用假的向量驗證排序與匯出。"""

import numpy as np

from conftest import make_video
from vsort import sidecar as sc
from vsort.analyze import UNKNOWN_AUTHOR_DIR, file_key
from vsort.style_author import StyleAuthorIndex, export_unknown


def _emb(cfg, video, vec):
    d = cfg.layout.workspace / "_cache" / "emb" / "siglip"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{file_key(video)}.npy", np.array(vec, dtype=np.float32))


def test_suggest_by_style(cfg):
    lib = cfg.layout.library
    for i, (author, vec) in enumerate([("A", [1, 0, 0]), ("A", [0.9, 0.1, 0]), ("B", [0, 1, 0]), ("C", [0, 0, 1])]):
        v = make_video(lib / author / f"v{i} [{author}][-][-][MMD][Dance].mp4", size=2048 + i)
        _emb(cfg, v, vec)
    q = make_video(cfg.layout.review / "q [-][-][-][MMD][Dance].mp4", size=4096)
    _emb(cfg, q, [0.95, 0.05, 0.02])
    sugg = StyleAuthorIndex(cfg).suggest(file_key(q))
    assert sugg[0][0] == "A"
    assert len({a for a, _ in sugg}) == len(sugg) <= 3


def test_export_unknown(cfg):
    v = make_video(cfg.layout.library / UNKNOWN_AUTHOR_DIR / "x [-][-][-][MMD][Dance].mp4")
    sc.write(v, {"info": {"title": "x"}, "original_name": "x.mp4", "style_authors": [["A", 0.8]]})
    out, n = export_unknown(cfg)
    assert n == 1 and "A（0.80）" in out.read_text(encoding="utf-8-sig")
