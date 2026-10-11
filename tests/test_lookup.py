"""網路查詢與作者別名：用假的 HTTP 回應（格式取自實測），不連網。"""

import json

import pytest
import yaml

from conftest import make_video
from vsort import sidecar as sc
from vsort.authors import accept_pending, update_authors
from vsort.cli import main
from vsort.config import load_config
from vsort.dictionary import Dictionary
from vsort.lookup import Lookup, make_enricher, weak_title
from vsort.net import NetError
from vsort.parser import Parser
from vsort.process import plan_process

IWARA = {"id": "Xy12Ab34Cd56Ef", "title": "Poison", "rating": "general",
         "user": {"id": "uuid-deltak", "name": "DeltaK", "username": "deltak"},
         "tags": [{"id": "koikatsu"}, {"id": "wuthering_waves"}], "createdAt": "2025-04-25T12:24:30.000Z"}
FANBOX = {"body": {"post": {"id": "10203040", "title": "mkc🌸と手繋ぎ散歩", "creatorId": "usagikobo",
                            "user": {"userId": "100200300", "name": "うさぎ工房"}, "tags": []}}}
FANTIA_HTML = '<meta name="csrf-token" content="tok123" />'
FANTIA = {"post": {"id": 3301234, "title": "【#38】エルフのおもてなし", "rating": "general",
                   "fanclub": {"id": 12345, "name": "ABC工房", "creator_name": "purinu"},
                   "tags": [{"name": "エルフのラドラル"}]}}


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _hit(self, url, kw):
        self.calls.append((url, kw))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise NetError("notfound", "HTTP 404")

    def json(self, url, **kw):
        return self._hit(url, kw)

    def text(self, url, **kw):
        return self._hit(url, kw)


@pytest.fixture
def online_cfg(cfg):
    cfg.data["lookup"] = {"enabled": True, "use_kemono": True, "kemono_domain": "kemono.cr", "mirrors": ["kemono.cr"],
                          "use_idm": False, "use_hints": False, "use_oreno3d": False}  # 測試不連真網站
    return cfg


def _enrich(cfg, routes, secrets=None):
    http = FakeHttp(routes)
    lk = Lookup(cfg, secrets=secrets or {}, http=http)
    dic = Dictionary(cfg.dictionary_dir)
    return make_enricher(lk, dic), Parser(dic), http


def test_iwara_lookup_and_cache(online_cfg):
    enrich, parser, http = _enrich(online_cfg, {"api.iwara.tv/video/Xy12Ab34Cd56Ef": IWARA})
    p = parser.parse("Iwara - Poison [Xy12Ab34Cd56Ef] [Source].mp4", ["iwara"])
    r = enrich(p, {})
    assert r.info["author"] == "DeltaK"
    assert r.info["works"] == ["鳴潮"] and r.info["anim_type"] == ["Koikatsu"]
    assert r.data["lookup"]["author_candidate"]["account_id"] == "uuid-deltak"
    enrich(p, {})
    assert len(http.calls) == 1  # 第二次用快取


def test_iwara_login_when_needed(online_cfg):
    calls = {"n": 0}

    class LoginHttp(FakeHttp):
        def json(self, url, **kw):
            if "video/" in url:
                calls["n"] += 1
                auth = (kw.get("headers") or {}).get("Authorization")
                if not auth:
                    raise NetError("notfound", "HTTP 404")  # 未登入看不到
                assert auth == "Bearer access-1"
                return IWARA
            return super().json(url, **kw)

    http = LoginHttp({"user/login": {"token": "refresh-1"}, "user/token": {"accessToken": "access-1"}})
    lk = Lookup(online_cfg, secrets={"iwara": {"email": "a@b.c", "password": "x"}}, http=http)
    info = lk.fetch("iwara", "Xy12Ab34Cd56Ef")
    assert info.via == "login" and info.author_name == "DeltaK"


def test_fanbox_public(online_cfg):
    enrich, parser, _ = _enrich(online_cfg, {"api.fanbox.cc/post.info?postId=10203040": FANBOX})
    p = parser.parse("251129_mkc.mp4", ["うさぎ工房", "10203040-2025-11-29-mkc"])
    r = enrich(p, {})
    assert r.info["author"] == "うさぎ工房"
    assert r.info["title"].startswith("mkc🌸")  # 檔名標題太短，改用網站標題


def test_fantia_guess_from_number_prefix(online_cfg):
    enrich, parser, _ = _enrich(online_cfg, {"fantia.jp/posts/3301234": FANTIA_HTML,
                                             "fantia.jp/api/v1/posts/3301234": FANTIA})
    p = parser.parse("3301234 エルフのラドラルちゃん_4K.mp4", ["未分類"])
    r = enrich(p, {})
    assert r.info["author"] == "purinu"
    assert "おもてなし" not in r.info.get("title", "")  # 檔名標題夠清楚，保留


def test_paid_falls_back_to_kemono_then_cookie(online_cfg):
    routes = {"kemono.cr/api/v1/fanbox/post/555": {"service": "fanbox", "artist_id": "9", "post_id": "555"},
              "kemono.cr/api/v1/fanbox/user/9/post/555": {"post": {"title": "鏡像標題", "id": "555"}},
              "kemono.cr/api/v1/fanbox/user/9/profile": {"name": "作者K"}}
    lk = Lookup(online_cfg, secrets={}, http=FakeHttp(routes))
    info = lk.fetch("fanbox", "555")
    assert info.via == "kemono" and info.title == "鏡像標題" and info.author_name == "作者K"

    # kemono 也沒有 → 用 cookie
    class CookieHttp(FakeHttp):
        def json(self, url, **kw):
            if "post.info" in url and "Cookie" in (kw.get("headers") or {}):
                return {"body": {"post": {"title": "會員限定", "user": {"userId": "1", "name": "作者C"}}}}
            return super().json(url, **kw)

    lk2 = Lookup(online_cfg, secrets={"cookies": {"fanbox": "FANBOXSESSID=x"}}, http=CookieHttp({}))
    info2 = lk2.fetch("fanbox", "777")
    assert info2.via == "cookie" and info2.author_name == "作者C"


def test_network_failure_goes_to_staging(online_cfg, tmp_path):
    make_video(online_cfg.layout.inbox / "iwara" / "Iwara - X [AAAAAAAAAAAAAA] [Source].mp4")
    enrich, parser, _ = _enrich(online_cfg, {"api.iwara.tv": NetError("retry", "連線失敗")})
    plan, counts = plan_process(online_cfg, enrichers=[enrich], parser=parser)
    assert counts == {"staging": 1}


def test_notfound_still_routes(online_cfg):
    make_video(online_cfg.layout.inbox / "Mochi" / "Iwara - X [AAAAAAAAAAAAAA] [Source].mp4")
    enrich, parser, _ = _enrich(online_cfg, {})
    plan, counts = plan_process(online_cfg, enrichers=[enrich], parser=parser)
    assert counts == {"refine": 1}
    assert "查無" in plan.ops[0].note


def test_weak_title():
    assert weak_title("mkc") and weak_title("MMKF0100") and weak_title("p1") and weak_title("bonus")
    assert not weak_title("がんばる天津風") and not weak_title("Gishinaki Atago")


def test_author_aliases_flow(online_cfg):
    """新作者自動加入；放在已知作者資料夾、帳號卻是新的 → 待確認；接受後下次用 ID 認人。"""
    lay = online_cfg.layout
    dic_dir = online_cfg.dictionary_dir
    (dic_dir / "authors.csv").write_text("標準名,網站,帳號ID,別名\nMochi,fanbox,111,モチ\n", encoding="utf-8-sig")

    def put(name, cand):
        v = make_video(lay.refine / name)
        sc.write(v, {"lookup": {"author_candidate": cand}})

    put("a.mp4", {"canonical": "DeltaK", "site": "iwara", "account_id": "uuid-b", "name": "DeltaK"})
    put("b.mp4", {"canonical": "Mochi", "site": "iwara", "account_id": "uuid-s", "name": "surume_mmd"})
    put("c.mp4", {"canonical": "モチ2", "site": "fantia", "account_id": "222", "name": "モチ2"})
    added, queued = update_authors(online_cfg)
    assert (added, queued) == (1, 2)
    pending = (dic_dir / "authors_pending.csv").read_text(encoding="utf-8-sig")
    assert "uuid-s" in pending and "222" in pending
    # 再跑一次不重複
    assert update_authors(online_cfg) == (0, 0)

    rows = (dic_dir / "authors_pending.csv").read_text(encoding="utf-8-sig").splitlines()
    rows = [("Y" + r) if "uuid-s" in r else ("N" + r) if ",222," in r else r for r in rows]
    (dic_dir / "authors_pending.csv").write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
    assert accept_pending(online_cfg) == (1, 1)
    d = Dictionary(dic_dir)
    assert d.author_by_account("iwara", "uuid-s") == "Mochi"
    assert d.author_by_account("iwara", "uuid-b") == "DeltaK"
    assert d.author_by_account("fantia", "222") is None
    assert update_authors(online_cfg) == (0, 0)  # 拒絕過的不再問
