from pathlib import Path

from vsort.naming import fit_path, sanitize_filename, unique_path


def test_illegal_chars_fullwidth():
    assert sanitize_filename('a:b?c*"d<e>f|g.MP4') == "a：b？c＊＂d＜e＞f｜g.mp4"


def test_illegal_chars_underscore():
    assert sanitize_filename("a:b/c.mp4", style="underscore") == "a_b_c.mp4"


def test_brackets_and_cjk_kept():
    name = "新年舞蹈 [某作者][原神][雷電將軍][MMD][Dance].mp4"
    assert sanitize_filename(name) == name


def test_trailing_dots_spaces_control():
    assert sanitize_filename("  abc\t\x01 def. . .mp4") == "abc def.mp4"


def test_reserved_and_empty():
    assert sanitize_filename("CON.mp4") == "CON_.mp4"
    assert sanitize_filename("  . .mp4") == "untitled.mp4"


def test_length_limit():
    out = sanitize_filename("字" * 300 + ".mp4", max_len=180)
    assert out == "字" * 180 + ".mp4"


def test_unique_path(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    taken = set()
    p1 = unique_path(tmp_path / "a.mp4", taken)
    p2 = unique_path(tmp_path / "a.mp4", taken)
    assert p1.name == "a (2).mp4"
    assert p2.name == "a (3).mp4"


def test_unique_case_insensitive(tmp_path):
    taken = set()
    unique_path(tmp_path / "A.mp4", taken)
    assert unique_path(tmp_path / "a.mp4", taken).name == "a (2).mp4"


def test_fit_path():
    p = Path("E:/" + "d" * 100) / ("x" * 200 + ".mp4")
    assert len(str(fit_path(p, 250))) <= 250
