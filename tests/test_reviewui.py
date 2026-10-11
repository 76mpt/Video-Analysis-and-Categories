"""人工確認網頁：起一個本機伺服器，實際打 API。"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from conftest import make_video
from vsort import sidecar as sc
from vsort.reviewui import Handler


def _server(cfg):
    handler = type("H", (Handler,), {"cfg": cfg})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_review_web_flow(cfg):
    v = make_video(cfg.layout.review / "x [-][-][-][3D][Scene].mp4")
    sc.write(v, {"info": {"title": "x", "anim_type": "3D", "category": "Scene"}, "reasons": ["不知道作者"],
                 "suggest_authors": ["Mochi"]})
    sc.preview_for(v).write_bytes(b"\xff\xd8fakejpeg")
    httpd, base = _server(cfg)
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "人工確認" in page
        items = json.load(urllib.request.urlopen(base + "/api/items"))["items"]
        assert items[0]["suggest_authors"] == ["Mochi"] and items[0]["preview"]
        assert urllib.request.urlopen(base + "/preview?name=" + urllib.parse.quote(v.name)).read().startswith(b"\xff\xd8")
        # 路徑穿越被擋
        code, _ = _post(base + "/api/confirm", {"name": "..\..\config.yaml", "info": {}})
        assert code == 404
        code, res = _post(base + "/api/confirm", {"name": v.name, "info": {
            "title": "x", "author": "Mochi", "works": "艦隊收藏", "characters": "不知火", "anim_type": "3D", "category": "Scene"}})
        assert code == 200, res
        dst = cfg.layout.library / "Mochi" / "x [Mochi][艦隊收藏][Shiranui][3D][Scene].mp4"
        assert dst.exists() and not v.exists()
        code, res = _post(base + "/api/undo", {})
        assert code == 200 and res["restored"] == 1
        assert v.exists() and not dst.exists()
    finally:
        httpd.shutdown()
