"""沒有網站 ID 時找來源（IDM → 檔名線索 → oreno3D）、iwara 刪片改用 oreno3D、pawchive 備援、secrets 防呆。
全部用假的物件，不連網、不讀登錄檔。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_video
from test_lookup import IWARA, FakeHttp
from vsort.dictionary import Dictionary
from vsort.lookup import Lookup, load_secrets, make_enricher
from vsort.net import NetError
from vsort.parser import Parser


@pytest.fixture
def lk_cfg(cfg):
    cfg.data["lookup"] = {"enabled": True, "use_kemono": True, "mirrors": ["kemono.cr", "pawchive.st"],
                          "use_idm": True, "use_hints": True, "use_oreno3d": True}
    return cfg


def _setup(cfg, routes, idm=None, hint=None, oreno=None):
    lk = Lookup(cfg, secrets={}, http=FakeHttp(routes))
    lk._idm = idm if idm is not None else False
    lk._hints = SimpleNamespace(search=lambda h: hint)
    lk._oreno = SimpleNamespace(find=lambda titles, author=None: (oreno, 1.0) if oreno else None)
    dic = Dictionary(cfg.dictionary_dir)
    return lk, make_enricher(lk, dic), Parser(dic)


def test_idm_first(lk_cfg, tmp_path):
    v = make_video(tmp_path / "MMD Puchi" / "tokio_dance.mp4")
    idm = SimpleNamespace(match=lambda path: SimpleNamespace(site="iwara", source_id="Xy12Ab34Cd56Ef",
                                                             account=None, how="size+name"))
    lk, enrich, parser = _setup(lk_cfg, {"api.iwara.tv/video/Xy12Ab34Cd56Ef": IWARA}, idm=idm)
    p = parser.parse(v.name, ["MMD Puchi"])
    r = enrich(p, {}, path=v)
    assert r.data["lookup"]["found_by"].startswith("IDM")
    assert r.info["author"] == "MMD Puchi" and "IDM" in r.note


def test_hints_when_no_idm(lk_cfg, tmp_path):
    v = make_video(tmp_path / "うさぎ工房" / "mkc.mp4")
    hint = SimpleNamespace(site="fanbox", source_id="10203040", account_id="100200300", reason="附件檔名相同",
                           score=1.0)
    fanbox = {"body": {"post": {"title": "mkc🌸と手繋ぎ", "user": {"userId": "100200300", "name": "うさぎ工房"}}}}
    lk, enrich, parser = _setup(lk_cfg, {"api.fanbox.cc/post.info?postId=10203040": fanbox}, hint=hint)
    r = enrich(parser.parse(v.name, ["うさぎ工房"]), {}, path=v)
    assert "檔名線索" in r.data["lookup"]["found_by"] and r.info["title"].startswith("mkc")


def test_oreno_when_nothing_else(lk_cfg, tmp_path):
    v = make_video(tmp_path / "x" / "Some Long Dance Title.mp4")
    mv = SimpleNamespace(iwara_id="Xy12Ab34Cd56Ef", title="Some Long Dance Title", author="DeltaK", tags=[],
                         origins=[], characters=[], published=None)
    lk, enrich, parser = _setup(lk_cfg, {"api.iwara.tv/video/Xy12Ab34Cd56Ef": IWARA}, oreno=mv)
    r = enrich(parser.parse(v.name, ["x"]), {}, path=v)
    assert r.data["lookup"]["found_by"] == "oreno3D 標題搜尋" and r.data["lookup"]["post_id"] == "Xy12Ab34Cd56Ef"


def test_deleted_iwara_uses_oreno(lk_cfg, tmp_path):
    mv = SimpleNamespace(iwara_id="AAAAAAAAAAAAAA", title="消えた動画", author="作者O", tags=["MMD"],
                         origins=["原神"], characters=["甘雨"], published="2024-01-01")
    lk, enrich, parser = _setup(lk_cfg, {"api.iwara.tv/video/AAAAAAAAAAAAAA": NetError("notfound", "HTTP 404")},
                                oreno=mv)
    p = parser.parse("Iwara - 消えた動画 [AAAAAAAAAAAAAA] [Source].mp4", ["iwara"])
    r = enrich(p, {})
    assert r.data["lookup"]["status"] == "ok" and r.data["lookup"]["via"] == "oreno3d"
    assert r.info["author"] == "作者O" and "甘雨" in r.info["characters"] and r.info["anim_type"] == ["MMD"]
    assert "iwara 已刪除" in r.note


def test_pawchive_after_kemono(lk_cfg):
    routes = {"api.fanbox.cc": NetError("notfound", "HTTP 404"),
              "kemono.cr": NetError("notfound", "HTTP 404"),
              "pawchive.st/api/v1/fanbox/user/9/post/555": {"post": {"id": "555", "title": "鏡像二號"}},
              "pawchive.st/api/v1/fanbox/user/9/profile": {"name": "作者P"}}
    lk = Lookup(lk_cfg, secrets={}, http=FakeHttp(routes))
    info = lk.fetch("fanbox", "555", artist="9")
    assert info.via == "pawchive" and info.author_name == "作者P"


def test_secrets_missing_space_is_repaired(tmp_path, capsys):
    p = tmp_path / "secrets.yaml"
    p.write_text("iwara:\n  email:a@b.c\n  password:xyz\ncookies:\n  fanbox: FANBOXSESSID=1\n"
                 "saucenao:\n  api_key:abc123\n", encoding="utf-8")
    s = load_secrets(p)
    assert s["iwara"] == {"email": "a@b.c", "password": "xyz"}
    assert s["saucenao"] == {"api_key": "abc123"}
    assert s["cookies"] == {"fanbox": "FANBOXSESSID=1"}
    assert "冒號後面要加一個空格" in capsys.readouterr().err


def test_secrets_broken_yaml(tmp_path):
    p = tmp_path / "secrets.yaml"
    p.write_text("iwara: [unclosed\n", encoding="utf-8")
    assert load_secrets(p) == {}


def test_site_skipped_after_repeated_failures(lk_cfg):
    http = FakeHttp({"api.iwara.tv": NetError("retry", "HTTP 429")})
    lk = Lookup(lk_cfg, secrets={}, http=http)
    for vid in ("A1", "A2", "A3"):
        with pytest.raises(NetError):
            lk.fetch("iwara", vid * 7)
    n = len(http.calls)
    with pytest.raises(NetError) as ei:
        lk.fetch("iwara", "A4" * 7)
    assert ei.value.kind == "retry" and len(http.calls) == n  # 第 4 次直接跳過，不再連線
