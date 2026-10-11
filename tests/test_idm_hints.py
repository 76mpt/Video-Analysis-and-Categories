from datetime import date

from vsort.hints import HintSearcher, Hints, parse_date, similarity
from vsort.idm import IdmIndex, entry_from_values, source_of


def _size(n: int) -> bytes:
    return n.to_bytes(8, "little")


def test_source_of_iwara_page():
    e = entry_from_values("1", {"Url0": "https://hime.iwara.tv/download?filename=x_Source.mp4&download=Iwara+-+t+%5BAbc%5D",
                                "owWPage": "https://www.iwara.tv/video/vmYtwDLUtf5c6Q/some-title",
                                "FileName": "x_Source.mp4", "FileSize": _size(5_000_000)})
    s = source_of(e)
    assert (s["site"], s["source_id"]) == ("iwara", "vmYtwDLUtf5c6Q")
    assert s["orig_name"].startswith("Iwara - t")


def test_source_of_fanbox_and_mirror():
    e = entry_from_values("2", {"Url0": "https://downloads.fanbox.cc/files/post/1442626/kfx.mp4",
                                "owWPage": "https://ohg58tte37.fanbox.cc/"})
    assert source_of(e)["source_id"] == "1442626"
    e = entry_from_values("3", {"Url0": "https://n3.kemono.cr/data/aa/bb/x.mp4?f=a.mp4",
                                "owWPage": "https://kemono.cr/patreon/user/11112222/post/33334444"})
    s = source_of(e)
    assert (s["site"], s["source_id"], s["account"]) == ("patreon", "33334444", "11112222")
    e = entry_from_values("4", {"Url0": "https://fantia.jp/x", "owWPage": "https://fantia.jp/posts/3560470"})
    assert source_of(e)["site"] == "fantia"


def test_index_matches_by_size_even_if_renamed(tmp_path):
    f = tmp_path / "renamed.mp4"
    f.write_bytes(b"x" * 1_200_000)
    e = entry_from_values("5", {"Url0": "https://fantia.jp/a.mp4", "owWPage": "https://fantia.jp/posts/123456",
                                "FileName": "orig.mp4", "FileSize": _size(1_200_000)})
    m = IdmIndex([e]).match(f)
    assert m and m.how == "size" and m.source_id == "123456"


def test_index_ignores_ambiguous_size(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x" * 1_200_000)
    es = [entry_from_values(str(i), {"Url0": "u", "owWPage": f"https://fantia.jp/posts/{1000 + i}",
                                     "FileName": f"o{i}.mp4", "FileSize": _size(1_200_000)}) for i in range(2)]
    assert IdmIndex(es).match(f) is None


def test_parse_date_and_similarity():
    assert parse_date("26.3.20_ヒミ") == date(2026, 3, 20)
    assert parse_date("2025-06-29 x") == date(2025, 6, 29)
    assert similarity("Luka & Gumi - Scream (Scene & Dance)", "Scream Luka Gumi") > 0.8
    assert similarity("さとり", "まったく違う") < 0.5


def test_score_exact_filename_wins():
    h = Hints(author="μ", titles=["VR ブラマジガールでShakingDance2"],
              stem="d597dab1_03__VR_ブラマジガールでShakingDance2_Extended_4Kp60fps",
              rest="03__VR_ブラマジガールでShakingDance2_Extended_4Kp60fps")
    post = {"id": "3261456", "title": "【VR】ブラマジガールでShakingDance2",
            "files": ["03__VR_ブラマジガールでShakingDance2_Extended_4Kp60fps.mp4"]}
    assert HintSearcher.score(post, h)[0] == 1.0


def test_offline_searcher_returns_none(tmp_path):
    hs = HintSearcher(tmp_path, online=False)
    assert hs.search(Hints(author="x", titles=["abc"], stem="abc", rest="abc")) is None


def test_imgsearch_sources_in():
    from vsort.imgsearch import sources_in
    urls = ["https://www.xnxx.com/video-1/x", "https://www.iwara.tv/video/AbCdEfGh1234/x",
            "https://kemono.cr/fanbox/user/1/post/99999"]
    assert sources_in(urls) == [("iwara", "AbCdEfGh1234"), ("fanbox", "99999")]
