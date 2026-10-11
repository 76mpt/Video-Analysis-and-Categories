"""Parsing of the four filename types, using realistic sample filenames."""

from pathlib import Path

import pytest

from vsort.config import PROJECT_ROOT
from vsort.dictionary import Dictionary
from vsort.naming import build_final_name, parse_final_name
from vsort.parser import CODE_NAME, CODE_SUBID, PLAIN, URL_ID, Parser
from vsort.process import info_from_parsed


@pytest.fixture(scope="module")
def parser():
    return Parser(Dictionary(PROJECT_ROOT / "dict"))


# (relative path, type, site, ID, creator, works, characters, animation type)
CASES = [
    # URL IDs: iwara downloader
    ("iwara/Iwara - Kuromaru Exercise-黒丸エクササイズ [aB3dE5fG7hJ9kL] [Source].mp4", URL_ID, "iwara", "aB3dE5fG7hJ9kL", None, [], [], None),
    ("iwara/Iwara - 申鶴 夏祭り【原神】 [Qw7Er9Ty1Ui3Op] [Source].mp4", URL_ID, "iwara", "Qw7Er9Ty1Ui3Op", None, ["原神"], ["申鶴"], None),
    ("iwara/Iwara - 【艦これ】摩耶様のダンス Maya dance practice [Zx2Cv4Bn6Mm8Ll] [Source].mp4", URL_ID, "iwara", "Zx2Cv4Bn6Mm8Ll", None, ["艦隊收藏"], ["摩耶"], None),
    ("iwara/Iwara - 符玄的舞台 [Po9Iu8Yt7Re6Wq] [Source].mp4", URL_ID, "iwara", "Po9Iu8Yt7Re6Wq", None, ["崩壞：星穹鐵道"], ["符玄"], None),
    ("iwara/Iwara - 【MMD】甘雨の“慰労会” （衣装Ver.） [Lk5Jh4Gf3Ds2Aa] [Source].mp4", URL_ID, "iwara", "Lk5Jh4Gf3Ds2Aa", None, ["原神"], ["甘雨"], "MMD"),
    ("iwara/Iwara - 嵐ちゃん１【艦これ】Arashi - Morning Routine [Mn1Bv2Cx3Zz4Qq] [Source].mp4", URL_ID, "iwara", "Mn1Bv2Cx3Zz4Qq", None, ["艦隊收藏"], ["嵐"], None),
    ("iwara/Iwara - mualani short 2 [Rt5Yu6Io7Pa8Ss] [Source].mp4", URL_ID, "iwara", "Rt5Yu6Io7Pa8Ss", None, ["原神"], ["瑪拉妮"], None),
    ("iwara/Iwara - イタコ姉さまのリンパマッサージ講座 [Df9Gh8Jk7Ll6Zx] [Source].mp4", URL_ID, "iwara", "Df9Gh8Jk7Ll6Zx", None, ["VOICEROID"], ["東北伊達子"], None),
    ("iwara/Iwara - Marin Kitagawa Party Tonight (Scene Dance) [Cv1Bn2Mq3We4Rr] [Source].mp4", URL_ID, "iwara", "Cv1Bn2Mq3We4Rr", None, ["戀上換裝娃娃"], ["喜多川海夢"], None),
    ("iwara/Iwara - Klukai - Bass Knight [Ty5Ui6Op7As8Dd] [Source].mp4", URL_ID, "iwara", "Ty5Ui6Op7As8Dd", None, ["少女前線2"], ["克魯凱"], None),
    ("iwara/Iwara - 【KK-ZZZ】某仙人的五行试炼评测记录 [Fg9Hj1Kl2Zx3Cc] [Source].mp4", URL_ID, "iwara", "Fg9Hj1Kl2Zx3Cc", None, ["絕區零"], [], "Koikatsu"),
    ("iwara/Iwara - Umamusume-ウマ娘-赛马娘 - トウカイテイオーのダンス (東海帝皇舞蹈) [Vb4Nm5Qw6Er7Tt] [Source].mp4", URL_ID, "iwara", "Vb4Nm5Qw6Er7Tt", None, ["賽馬娘"], ["東海帝皇"], None),
    ("iwara/Iwara - ぽじお式シスター ジェントルマン [yu8io9pa1sd2fg3h] [Source].mp4", URL_ID, "iwara", "yu8io9pa1sd2fg3h", None, [], [], None),
    ("iwara/Iwara - Astra Yao – Rock Right Now [Jk4Lz5Xc6Vb7Nn] [Source].mp4", URL_ID, "iwara", "Jk4Lz5Xc6Vb7Nn", None, ["絕區零"], ["耀嘉音"], None),
    ("用/HMV/Iwara - 【HMV】Big Juicy Party [Mq8We9Rt1Yu2Ii] [Source].mp4", URL_ID, "iwara", "Mq8We9Rt1Yu2Ii", None, [], [], "2D"),
    ("x/https___www.iwara.tv_video_ZkDvNqbXo4Uu8a_some-title.mp4", URL_ID, "iwara", "ZkDvNqbXo4Uu8a", "x", [], [], None),
    ("x/fantia.jp_posts_1234567.mp4", URL_ID, "fantia", "1234567", "x", [], [], None),
    ("x/patreon.com_posts_cool-video-98765432.mp4", URL_ID, "patreon", "98765432", "x", [], [], None),
    ("x/fanbox.cc_@artist_posts_7654321.mp4", URL_ID, "fanbox", "7654321", "x", [], [], None),
    ("うさぎ工房/500(250709-260228)/10203040-2025-11-29-mkc🌸と手繋ぎ【フル尺】/251129_mkc.mp4", URL_ID, "fanbox", "10203040", "うさぎ工房", [], [], None),
    # random code + name (common with fanbox downloaders)
    ("Mochi/88aa854e_海風と山風のお茶会.mp4", CODE_NAME, None, None, "Mochi", ["艦隊收藏"], ["海風", "山風"], None),
    ("Mochi/1efc1688_ゆっくり休む不知火.mp4", CODE_NAME, None, None, "Mochi", ["艦隊收藏"], ["不知火"], None),
    ("kirin/kirin 12/678b8ac3_リン_Sour式__JUMP_UP.mp4", CODE_NAME, None, None, "kirin", ["VOCALOID"], ["鏡音鈴"], "MMD"),
    ("kirin/10/96ced205_クラウンピース_早苗_Love_Me_If_You_Can_通常版.mp4", CODE_NAME, None, None, "kirin", ["東方Project"], ["克勞恩皮絲", "東風谷早苗"], None),
    ("kirin/16b61fd7_チェシャー_Conqueror.mp4", CODE_NAME, None, None, "kirin", ["碧藍航線"], ["柴郡"], None),
    ("エビ/12/666c475a_今汐_ILLELLA.mp4", CODE_NAME, None, None, "エビ", ["鳴潮"], ["今汐"], None),
    ("エビ/12/2d03b5b5_ジャンヌオルタ_wiggle.mp4", CODE_NAME, None, None, "エビ", ["FGO"], ["貞德Alter"], None),
    ("エビ/9ee48ccf_ジャンヌ_紫式部_デメテルだんす.mp4", CODE_NAME, None, None, "エビ", ["FGO"], ["紫式部"], None),
    ("のんびり村民/4912bd2a_osario.mp4", CODE_NAME, None, None, "のんびり村民", [], [], None),
    ("Haru/fa01ceed_Ray_ダイナミックダンス_頼光_01_A_ノーマル_効果音付き_2160p.mp4", CODE_NAME, None, None, "Haru", ["FGO"], ["源賴光"], None),
    ("未分類/3301234 エルフのラドラルちゃん_4K.mp4", CODE_NAME, None, None, None, [], [], None),
    # random code + sub ID
    ("x/ab12cd34_3456789.mp4", CODE_SUBID, None, "3456789", "x", [], [], None),
    # filename only
    ("mikanman/komachi_jump_4k.mp4", PLAIN, None, None, "mikanman", ["東方Project"], ["小野塚小町"], None),
    ("mikanman/yuukarin_gokuraku_4k.mp4", PLAIN, None, None, "mikanman", ["東方Project"], ["風見幽香"], None),
    ("mikanman/4K_風見幽香で極楽浄土.mp4", PLAIN, None, None, "mikanman", ["東方Project"], ["風見幽香"], None),
    ("Mochi/[KanColle]がんばる天津風.mp4", PLAIN, None, None, "Mochi", ["艦隊收藏"], ["天津風"], None),
    ("Mochi/Umikaze Picnic.mp4", PLAIN, None, None, "Mochi", ["艦隊收藏"], ["海風"], None),
    ("The Studio/Tier 3/Tier 3 Variants/Haku - Golden (Scene Remix)/Golden_4K_Dance Only_Lip OFF.mp4", PLAIN, None, None, "The Studio", ["VOCALOID"], ["弱音白"], None),
    ("The Studio/Tier 3/Tier 3 Variants/MIku & Teto - Wiggle Wiggle (Scene & Dance)/Wiggle Wiggle_4K_Dance Only.mp4", PLAIN, None, None, "The Studio", ["VOCALOID"], ["初音未來", "重音Teto"], None),
    ("hikaru/Sigrika - Thumbs Up 4K/Sigrika - Thumbs Up 4K.mp4", PLAIN, None, None, "hikaru", [], [], None),
    ("未整理/【Blue Archive】Pump_Up_the_Jam_4K.mp4", PLAIN, None, None, None, ["蔚藍檔案"], [], None),
    ("未整理/托帕-Party Tonight Dance.mp4", PLAIN, None, None, None, ["崩壞：星穹鐵道"], ["托帕"], None),
    ("未整理/(JP)天龍ちゃんの鬼ごっこ(4K60fps).mp4", PLAIN, None, None, None, ["艦隊收藏"], ["天龍"], None),
    ("未分類/08C.【Genshin Impact】 Eula's afternoon 2k60FPS +Sound.mp4", PLAIN, None, None, None, ["原神"], ["優菈"], None),
    ("未分類/ときのそら - ELECT (Youtube風).mp4", PLAIN, None, None, None, ["Hololive"], ["時乃空"], None),
    ("未分類/【kanata MMD】 RAGE DANCE 4K.mp4", PLAIN, None, None, None, [], [], "MMD"),
    ("用/241030_호두+푸리나_메스머라이저_FHD.mp4", PLAIN, None, None, None, ["原神"], ["胡桃", "芙寧娜"], None),
    ("touhou/萃香ちゃんと宴会/suika_party.mp4", PLAIN, None, None, None, ["東方Project"], ["伊吹萃香"], None),
    ("250209 - ロビン - 踊/ロビン - 踊 (iwara用スクリーン).mp4", PLAIN, None, None, None, ["崩壞：星穹鐵道"], ["知更鳥"], None),
    ("mmdclips31/Gishinaki Atago.mp4", PLAIN, None, None, "mmdclips31", [], ["愛宕"], None),
    ("Neo_QOS/33.Marin QOS Scene Dance.mp4", PLAIN, None, None, "Neo_QOS", ["戀上換裝娃娃"], ["喜多川海夢"], None),
]


@pytest.mark.parametrize("rel,typ,site,sid,author,works,chars,anim", CASES, ids=[c[0][-40:] for c in CASES])
def test_real_filenames(parser, rel, typ, site, sid, author, works, chars, anim):
    parts = Path(rel).parts
    p = parser.parse(parts[-1], list(parts[:-1]))
    assert p.name_type == typ
    assert p.site == site
    assert p.source_id == sid
    assert p.author == author
    assert sorted(p.works) == sorted(works)
    assert p.characters == chars
    assert p.anim_type == ([anim] if anim else [])
    assert p.title and "[Source]" not in p.title and "4K" not in p.title


def test_no_false_positive_lymph(parser):
    p = parser.parse("リンパマッサージ.mp4")
    assert "鏡音鈴" not in p.characters


def test_titles(parser):
    assert parser.parse("Iwara - 申鶴 夏祭り【原神】 [Qw7Er9Ty1Ui3Op] [Source].mp4").title == "申鶴 夏祭り"
    assert parser.parse("88aa854e_お茶会で_海風.mp4").title == "お茶会で 海風"
    assert parser.parse("4K 夏 (3).mp4").title == "夏"
    assert parser.parse("241129 - 名塩ノア - ピチカートドロップス.mp4").date == "2024-11-29"


HINTS = {"Mix": [r"scene ?(?:&|and)? ?dance", r"scene remix"], "Scene": [r"\bscene\b"], "Dance": [r"dance only"]}


@pytest.fixture(scope="module")
def hinted(parser):
    return Parser(parser.dic, {"category_hints": HINTS})


def test_category_hints(parser, hinted):
    assert parser.parse("Wiggle Wiggle_4K_Dance Only.mp4").category_hint == "Dance"
    assert hinted.parse("Marin Party Tonight (Scene Dance).mp4").category_hint == "Mix"
    assert hinted.parse("komachi_jump scene.mp4").category_hint == "Scene"
    assert hinted.parse("Wiggle Wiggle_4K_Dance Only.mp4").category_hint == "Dance"


def test_final_name_roundtrip(parser):
    p = parser.parse("申鶴 夏祭り【原神】.mp4", ["作者[A]"])
    name = build_final_name(info_from_parsed(p), ".MP4")
    assert name == "申鶴 夏祭り [作者［A］][原神][申鶴][-][-].mp4"
    back = parse_final_name(name)
    assert back["author"] == "作者［A］" and back["characters"] == ["申鶴"] and back["category"] is None


def test_final_name_truncates_title_only():
    info = {"title": "長" * 300, "author": "作者", "works": ["原神"], "characters": ["甘雨"],
            "anim_type": "MMD", "category": "Dance"}
    name = build_final_name(info, ".mp4", max_len=100)
    assert name.endswith(" [作者][原神][甘雨][MMD][Dance].mp4")
    assert len(name) <= 104


def test_hint_prefers_filename_over_folder(hinted):
    p = hinted.parse("Wiggle Wiggle_4K_Dance Only.mp4", ["The Studio", "MIku & Teto - Wiggle Wiggle (Scene & Dance)"])
    assert p.category_hint == "Dance"
    q = hinted.parse("Golden_4K.mp4", ["The Studio", "Haku - Golden (Scene Remix)"])
    assert q.category_hint == "Mix"


def test_parser_multi_anim(parser):
    assert parser.parse("Lamb [MMD] blender render.mp4").anim_type == ["MMD", "3D"]


def test_koikatsu_vam(parser):
    assert parser.parse("コイカツ 甘雨.mp4").anim_type == ["Koikatsu"]
    assert parser.parse("Tifa VAM 4K.mp4").anim_type == ["VAM"]
    from vsort.naming import anim_list
    assert anim_list("戀活,無") == ["Koikatsu", "Audio"]
