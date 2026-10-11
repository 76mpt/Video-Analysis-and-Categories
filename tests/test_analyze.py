"""AI 分析與人工確認：用假的模型（不需要 GPU 或下載），驗證彙整、合併、分流與還原。"""

import csv

from PIL import Image

from conftest import make_video
from vsort import sidecar as sc
from vsort.analyze import DEFAULTS, AIResult, Analyzer, danbooru_to_names, decide, plan_analyze, summarize
from vsort.cli import main
from vsort.dictionary import Dictionary
from vsort.journal import Run, execute
from vsort.review import csv_path, export, plan_import

CHARS = {"hatsune_miku", "kasane_teto", "shiranui_(kancolle)", "foo_bar_(some_game)"}


def frames(n, **tags):
    return [dict(tags) for _ in range(n)]


def test_summarize_categories():
    dance = summarize(frames(16, dancing=0.9, hatsune_miku=0.95), CHARS, DEFAULTS)
    assert dance.category == "Dance" and dance.category_sure
    assert dance.characters == [{"tag": "hatsune_miku", "prob": 0.95, "frames": 16}]

    scene = summarize(frames(16, scene_tag=0.9), CHARS, DEFAULTS)
    assert scene.category == "Scene" and scene.category_sure

    mix = summarize(frames(8, dancing=0.9) + frames(8, scene_tag=0.9), CHARS, DEFAULTS)
    assert mix.category == "Mix" and mix.category_sure

    one_bad = summarize(frames(15, dancing=0.9) + frames(1, scene_tag=0.9), CHARS, DEFAULTS)
    assert one_bad.category == "Dance" and one_bad.category_sure  # 16 張裡只有 1 張，仍算跳舞

    border = summarize(frames(14, dancing=0.9) + frames(2, scene_tag=0.9), CHARS, DEFAULTS)
    assert border.category == "Mix" and not border.category_sure  # 12.5% 剛過門檻 → 可能誤判


def test_character_needs_two_frames():
    r = summarize(frames(15) + frames(1, kasane_teto=0.99), CHARS, DEFAULTS)
    assert r.characters == []


def test_danbooru_names(cfg):
    dic = Dictionary(cfg.dictionary_dir)
    assert danbooru_to_names("shiranui_(kancolle)", dic) == ("不知火", "艦隊收藏")
    assert danbooru_to_names("hatsune_miku", dic) == ("初音未來", "VOCALOID")
    assert danbooru_to_names("foo_bar_(some_game)", dic) == ("Foo Bar", "Some Game")


def test_decide_reasons(cfg):
    dic = Dictionary(cfg.dictionary_dir)
    sure_ai = AIResult(frames=16, category="Dance", category_sure=True,
                       characters=[{"tag": "hatsune_miku", "prob": 0.97, "frames": 12}],
                       style={"MMD": 0.9, "3D": 0.08, "2D": 0.02})
    info, sure, reasons = decide({"info": {"title": "t", "author": "The Studio"}}, sure_ai, dic, DEFAULTS)
    assert sure and not reasons
    assert info["characters"] == ["Hatsune Miku"] and info["works"] == ["VOCALOID"] and info["anim_type"] == ["MMD"]
    assert info["characters_i18n"][0] == {"en": "Hatsune Miku", "zh": "初音未來", "ja": "初音ミク"}

    # 檔名寫 Scene Dance，AI 卻說純跳舞 → 可能誤判
    _, sure2, reasons2 = decide({"info": {"author": "x"}, "parsed": {"category_hint": "Mix"}}, sure_ai, dic, DEFAULTS)
    assert not sure2 and any("檔名" in r for r in reasons2)

    # 沒作者、動畫類型不確定
    vague = AIResult(frames=16, category="Scene", category_sure=True, style={"MMD": 0.45, "3D": 0.5, "2D": 0.05})
    info3, sure3, reasons3 = decide({"info": {"title": "t"}}, vague, dic, DEFAULTS)
    assert not sure3 and info3["anim_type"] == ["3D"]
    assert not any("作者" in r for r in reasons3) and any("動畫類型" in r for r in reasons3)


class FakeTagger:
    def __init__(self, per_video):
        self.per_video = per_video
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return self.per_video[images[0].info["video"]]

    def character_set(self):
        return CHARS


class FakeStyle:
    def __call__(self, images):
        return {"MMD": 0.9, "3D": 0.08, "2D": 0.02}


def fake_extract(video, n):
    img = Image.new("RGB", (64, 64), (200, 100, 50))
    img.info["video"] = video.name.split(" [")[0]
    return [(float(i), img) for i in range(n)]


def _setup(cfg):
    lay = cfg.layout
    a = make_video(lay.refine / "Wiggle [The Studio][-][-][-][-].mp4")
    sc.write(a, {"info": {"title": "Wiggle", "author": "The Studio", "works": [], "characters": []},
                 "parsed": {"category_hint": "Dance"}})
    b = make_video(lay.identify / "untitled [-][-][-][-][-].mp4")
    sc.write(b, {"info": {"title": "untitled"}})
    tagger = FakeTagger({"Wiggle": frames(16, dancing=0.9, hatsune_miku=0.97, kasane_teto=0.9),
                         "untitled": frames(16, scene_tag=0.9, **{"shiranui_(kancolle)": 0.7})})
    return a, b, Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())


def test_plan_analyze_routes_and_undo(cfg):
    a, b, analyzer = _setup(cfg)
    plan, counts = plan_analyze(cfg, analyzer)
    assert counts == {"library": 1, "review": 1}
    run = Run.create(cfg.layout, "analyze", plan.ops)
    execute(run)
    lib = list(cfg.layout.library.rglob("*.mp4"))
    assert [p.relative_to(cfg.layout.library).as_posix() for p in lib] == \
        ["The Studio/Wiggle [The Studio][VOCALOID][Hatsune Miku,Kasane Teto][MMD][Dance].mp4"]
    rev = list(cfg.layout.review.glob("*.mp4"))
    assert rev[0].name == "untitled [-][艦隊收藏][Shiranui][MMD][Scene].mp4"
    assert sc.preview_for(rev[0]).exists()  # 附預覽圖
    data = sc.read(rev[0])
    assert data["certainty"] == "可能誤判" and data["reasons"]
    # 快取：再規劃一次不會重跑模型
    calls = analyzer.tagger.calls
    plan_analyze(cfg, analyzer)
    assert analyzer.tagger.calls == calls

    # 還原
    assert main(["--config", str(cfg.path), "undo", "--apply"]) == 0
    assert a.exists() and b.exists()
    assert sc.read(a)["info"]["author"] == "The Studio" and "ai" not in sc.read(a)
    assert not list(cfg.layout.review.glob("*.vsort.jpg"))


def test_review_import(cfg):
    a, b, analyzer = _setup(cfg)
    plan, _ = plan_analyze(cfg, analyzer)
    execute(Run.create(cfg.layout, "analyze", plan.ops))
    path, n = export(cfg)
    assert n == 1
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    rows[1][0] = "Y"
    rows[1][3] = "Mochi"       # 補上作者
    rows[1][7] = "mix"          # 改類別（大小寫不拘）
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    export(cfg)  # 重新匯出要保留你的修改
    with open(path, encoding="utf-8-sig", newline="") as fh:
        assert list(csv.reader(fh))[1][3] == "Mochi"
    plan2 = plan_import(cfg)
    assert len(plan2.ops) == 1
    assert main(["--config", str(cfg.path), "review", "--import", "--apply"]) == 0
    moved = list((cfg.layout.library / "Mochi").glob("*.mp4"))
    assert moved[0].name == "untitled [Mochi][艦隊收藏][Shiranui][MMD][Mix].mp4"
    assert sc.read(moved[0])["certainty"] == "人工確認"
    assert not sc.preview_for(moved[0]).exists()
    _, n2 = export(cfg)
    assert n2 == 0


def test_library_refill(cfg):
    """F22：04 裡缺類別的檔案也會被重新分析。"""
    lay = cfg.layout
    v = make_video(lay.library / "The Studio" / "Wiggle [The Studio][VOCALOID][Hatsune Miku][MMD][-].mp4")
    tagger = FakeTagger({"Wiggle": frames(16, dancing=0.9, hatsune_miku=0.97)})
    an = Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())
    plan, counts = plan_analyze(cfg, an)
    assert not plan.ops
    plan, counts = plan_analyze(cfg, an, include_library=True)
    assert counts["library"] == 1
    assert plan.ops[0].dst.endswith("Wiggle [The Studio][VOCALOID][Hatsune Miku][MMD][Dance].mp4")


def test_danbooru_romaji_full_name(cfg):
    dic = Dictionary(cfg.dictionary_dir)
    assert danbooru_to_names("onozuka_komachi", dic) == ("小野塚小町", "東方Project")
    assert danbooru_to_names("maya_kai_ni_(kancolle)", dic) == ("摩耶", "艦隊收藏")


def test_author_style_prior(cfg):
    lay = cfg.layout
    for i in range(3):
        make_video(lay.library / "mikanman" / f"v{i} [mikanman][東方Project][-][MMD][Dance].mp4")
    from vsort.analyze import author_styles
    styles = author_styles(cfg)
    assert styles == {"mikanman": ["MMD"]}
    dic = Dictionary(cfg.dictionary_dir)
    ai = AIResult(frames=16, category="Dance", category_sure=True, style={"MMD": 0.49, "3D": 0.51, "2D": 0.0},
                  characters=[{"tag": "onozuka_komachi", "prob": 0.99, "frames": 13}])
    info, sure, reasons = decide({"info": {"title": "k", "author": "mikanman", "characters": ["小野塚小町"],
                                           "works": ["東方Project"]}}, ai, dic, DEFAULTS, styles)
    assert sure and info["anim_type"] == ["MMD"] and info["characters"] == ["Onozuka Komachi"]


def test_unknown_author_goes_to_unknown_folder(cfg):
    """不知道作者不算可能誤判、也不依角色猜作者；其他欄位有把握就歸到 04 的 作者未知。"""
    lay = cfg.layout
    make_video(lay.library / "Mochi" / "a [Mochi][艦隊收藏][不知火][3D][Scene].mp4")
    dic = Dictionary(cfg.dictionary_dir)
    ai = AIResult(frames=16, category="Scene", category_sure=True, style={"3D": 0.9, "MMD": 0.1, "2D": 0},
                  characters=[{"tag": "shiranui_(kancolle)", "prob": 0.95, "frames": 10}])
    info, sure, reasons = decide({"info": {"title": "x"}}, ai, dic, DEFAULTS, {})
    assert sure and not info.get("author") and not reasons

    v = make_video(lay.identify / "x [-][-][-][-][-].mp4")
    sc.write(v, {"info": {"title": "x"}})
    tagger = FakeTagger({"x": frames(16, scene_tag=0.9, **{"shiranui_(kancolle)": 0.95})})
    an = Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())
    plan, counts = plan_analyze(cfg, an)
    op = next(o for o in plan.ops if o.src == str(v))
    assert "作者未知" in op.dst and op.sidecar["suggest_authors"] == []


def test_reverse_search_clues(cfg):
    """以圖搜圖（Yandex）：只對作者未知且查詢失敗的影片，取 2 張畫面，結果只當線索記錄。"""
    from vsort.analyze import file_key
    calls = []

    class FakeYandex:
        def search(self, image, key):
            assert image.exists()
            calls.append(key)
            return [{"title": "Shiranui dance MMD", "url": "https://www.iwara.tv/video/ABCDEFGHIJKL12/x"},
                    {"title": "repost", "url": "https://example.com/v/1"}]

    a, b, analyzer = _setup(cfg)
    fdir = cfg.layout.workspace / "_cache" / "frames" / file_key(b)
    fdir.mkdir(parents=True)
    for i in range(16):
        Image.new("RGB", (8, 8)).save(fdir / f"{i:02d}.jpg")
    plan, _ = plan_analyze(cfg, analyzer, searcher=FakeYandex())
    op = next(o for o in plan.ops if "untitled" in o.src)
    rs = op.sidecar["reverse_search"]
    assert rs["sources"] == [["iwara", "ABCDEFGHIJKL12"]] and "Shiranui dance MMD" in rs["titles"]
    assert len(calls) == 2 and not op.sidecar.get("info", {}).get("author")  # 不自動填作者
    assert not any("以圖搜圖" in r for r in op.sidecar["reasons"])
    op_a = next(o for o in plan.ops if "Wiggle" in o.src)  # 已知作者的不查
    assert "reverse_search" not in op_a.sidecar


def test_multilabel_style_classifier(cfg):
    dic = Dictionary(cfg.dictionary_dir)
    ai = AIResult(frames=16, category="Dance", category_sure=True, style_kind="classifier",
                  style={"MMD": 0.93, "3D": 0.88, "2D": 0.02},
                  characters=[{"tag": "hatsune_miku", "prob": 0.97, "frames": 12}])
    info, sure, reasons = decide({"info": {"title": "t", "author": "A"}}, ai, dic, DEFAULTS)
    assert info["anim_type"] == ["MMD", "3D"] and sure
    ai.style = {"MMD": 0.6, "3D": 0.3, "2D": 0.02}
    info, sure, reasons = decide({"info": {"title": "t", "author": "A"}}, ai, dic, DEFAULTS)
    assert info["anim_type"] == ["MMD"] and not sure


def test_multilabel_filename_roundtrip():
    from vsort.naming import build_final_name, parse_final_name
    info = {"title": "Golden", "author": "X", "works": [], "characters": [], "anim_type": ["3D", "MMD"],
            "category": "Dance"}
    name = build_final_name(info, ".mp4")
    assert name == "Golden [X][-][-][MMD,3D][Dance].mp4"
    assert parse_final_name(name)["anim_type"] == ["MMD", "3D"]
    assert parse_final_name("a [X][-][-][-][-].mp4")["anim_type"] == []


def test_describer_breaks_category_tie(cfg):
    a, b, analyzer = _setup(cfg)

    class FakeDescriber:
        def describe(self, imgs):
            return {"model": "fake", "text": "two characters dancing", "json": {"category": "Mix"}}

    analyzer.opts["describe_model"] = "fake"
    analyzer._describer = FakeDescriber()
    # 讓 Wiggle 的類別落在門檻附近：14 張跳舞＋2 張場景 → Mix 但不確定
    analyzer.tagger.per_video["Wiggle"] = frames(14, dancing=0.9, hatsune_miku=0.97) + frames(2, scene_tag=0.9, hatsune_miku=0.97)
    sc.write(a, {"info": {"title": "Wiggle", "author": "The Studio"}})
    plan, counts = plan_analyze(cfg, analyzer)
    op = next(o for o in plan.ops if "Wiggle" in o.src)
    assert op.sidecar["certainty"] == "已確定", op.sidecar["reasons"]
    assert op.sidecar["description"]["text"]


def test_mmd_lenient_thresholds(cfg):
    from vsort.analyze import style_labels
    opts = dict(DEFAULTS)
    # MMD 0.35（低於一般門檻 0.5，但高於 MMD 的 0.3）→ 照樣標上，而且不算不確定
    labels, sure = style_labels(AIResult(style_kind="classifier",
                                         style={"MMD": 0.35, "Koikatsu": 0.9, "VAM": 0.01, "3D": 0.05, "2D": 0.0, "Audio": 0.0}), opts)
    assert labels == ["MMD", "Koikatsu"] and sure
    # 3D 在門檻附近 → 不確定
    labels, sure = style_labels(AIResult(style_kind="classifier",
                                         style={"MMD": 0.9, "Koikatsu": 0.0, "VAM": 0.0, "3D": 0.45, "2D": 0.0, "Audio": 0.0}), opts)
    assert labels == ["MMD"] and not sure
    # 非 MMD 的影片：MMD 機率很低（≤ 0.1）就是明顯沒有，不算不確定；0.15 離 MMD 門檻 0.3 不夠遠才算不確定
    labels, sure = style_labels(AIResult(style_kind="classifier",
                                         style={"MMD": 0.07, "Koikatsu": 0.97, "VAM": 0.0, "3D": 0.09, "2D": 0.0, "Audio": 0.0}), opts)
    assert labels == ["Koikatsu"] and sure
    labels, sure = style_labels(AIResult(style_kind="classifier",
                                         style={"MMD": 0.15, "Koikatsu": 0.97, "VAM": 0.0, "3D": 0.01, "2D": 0.0, "Audio": 0.0}), opts)
    assert labels == ["Koikatsu"] and not sure


def test_category_from_qwen(cfg):
    a, b, analyzer = _setup(cfg)

    class FakeQwen:
        def describe(self, imgs):
            return {"model": "qwen3vl", "text": "dancing then scene", "json": {"category": "Mix"}}

    analyzer.opts["category_model"] = "qwen3vl"
    analyzer._describers = {"qwen3vl": FakeQwen()}
    r = analyzer.run(a)  # 畫面標籤說 Dance，Qwen 說 Mix
    assert r.category == "Mix" and r.category_tags == "Dance" and not r.category_sure
    info, sure, reasons = decide({"info": {"title": "t", "author": "A"}}, r, Dictionary(cfg.dictionary_dir), analyzer.opts)
    assert not sure and any("Qwen3-VL 判斷 Mix" in x for x in reasons)
    r2 = analyzer.run(b)  # 畫面標籤說 Scene、Qwen 說 Mix → 以 Qwen 為準但不確定
    assert r2.category_tags == "Scene" and r2.category == "Mix" and not r2.category_sure


def test_category_classifier_roundtrip(tmp_path):
    import json as _json
    from vsort.category_train import CategoryClassifier, features
    x = features(0.9, 0.0, 0.9, {"category": "Scene", "scene_frames": 8})
    assert x == [0.9, 0.0, 0.9, 0.0, 0.0, 1.0, 1.0]
    assert features(0.0, 0.5, 0.0, None)[-1] == 0.5  # Qwen 沒回答時用中間值
    p = tmp_path / "c.json"
    p.write_text(_json.dumps({"classes": ["Dance", "Mix", "Scene"], "intercept": [0, 0, 0],
                              "coef": [[-5, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0], [5, 0, 0, 0, 0, 3, 0]]}))
    probs = CategoryClassifier(p).predict(x)
    assert max(probs, key=probs.get) == "Scene" and abs(sum(probs.values()) - 1) < 1e-9


def test_candidates_for_review(cfg):
    from vsort.analyze import candidates_for
    dic = Dictionary(cfg.dictionary_dir)
    per_frame = frames(8, hatsune_miku=0.9) + frames(8, **{"kasane_teto": 0.3, "shiranui_(kancolle)": 0.15})
    r = summarize(per_frame, CHARS, DEFAULTS)
    assert [c["tag"] for c in r.character_candidates] == ["hatsune_miku", "kasane_teto", "shiranui_(kancolle)"]
    r.category_probs = {"Scene": 0.2, "Mix": 0.5, "Dance": 0.3}
    r.style = {"MMD": 0.7, "Koikatsu": 0.4}
    c = candidates_for(r, dic, [["A", 0.8]])
    assert list(c["category"]) == ["Mix", "Dance", "Scene"]
    assert [x[0] for x in c["characters"]] == ["初音未來", "重音Teto", "不知火"]
    assert c["works"][0][0] == "VOCALOID" and c["authors"] == [["A", 0.8]]


def test_danbooru_costume_not_work(cfg):
    dic = Dictionary(cfg.dictionary_dir)
    assert danbooru_to_names("nekomata_okayu_(1st_costume)", dic) == ("貓又小粥", "Hololive")
    assert danbooru_to_names("pekomon_(usada_pekora)", dic)[1] is None


def test_works_not_required(cfg):
    """作品不影響自動歸檔；沒有作品時填信心最高的候選。"""
    dic = Dictionary(cfg.dictionary_dir)
    ai = AIResult(frames=16, category="Dance", category_sure=True, style={"MMD": 0.9}, style_kind="zeroshot",
                  characters=[{"tag": "hatsune_miku", "prob": 0.97, "frames": 12}],
                  character_candidates=[{"tag": "hatsune_miku", "prob": 0.97, "frames": 12}])
    info, sure, reasons = decide({"info": {"title": "t", "author": "A", "characters": ["原創角色"]}}, ai, dic, DEFAULTS)
    assert sure and "VOCALOID" in info["works"]
    info2, sure2, reasons2 = decide({"info": {"title": "t", "author": "A"}},
                                    AIResult(frames=16, category="Dance", category_sure=True, style={"MMD": 0.9}),
                                    dic, DEFAULTS)
    assert not sure2 and reasons2 == ["不知道角色"]


def test_series_reuses_confirmed_characters(cfg):
    """同作者、同系列（標題相近）已歸檔過的角色，新影片直接沿用。"""
    lay = cfg.layout
    make_video(lay.library / "The Studio" / "Wiggle Wiggle Full Ver [The Studio][VOCALOID][初音未來,重音Teto][MMD][Mix].mp4")
    v = make_video(lay.refine / "Wiggle Wiggle Dance Only [The Studio][-][-][-][-].mp4", size=4096)
    sc.write(v, {"info": {"title": "Wiggle Wiggle Dance Only", "author": "The Studio"}})
    tagger = FakeTagger({"Wiggle Wiggle Dance Only": frames(16, dancing=0.9)})  # AI 認不出角色
    an = Analyzer(cfg, dict(DEFAULTS, category_model="tags"), tagger=tagger, extractor=fake_extract, style=FakeStyle())
    plan, counts = plan_analyze(cfg, an)
    op = next(o for o in plan.ops if o.src == str(v))
    assert op.sidecar["info"]["characters"] == ["Hatsune Miku", "Kasane Teto"]
    assert op.sidecar["series_from"].startswith("Wiggle Wiggle Full Ver")
    assert op.sidecar["certainty"] == "已確定"


def test_fallback_second_stage_only_when_unsure(cfg):
    """類別信心不足時才加跑備援版，再用第二段分類器判斷。"""
    a, b, analyzer = _setup(cfg)
    analyzer.opts.update(category_model="combined", category_sure_prob=0.8, category_second_stage=True)
    calls = []

    class Clf:
        def __init__(self, fs, probs):
            self.feature_set, self.probs = fs, probs

        def predict(self, x):
            return self.probs

    analyzer._catclf = Clf("tags+frames_v1+video_v1", {"Dance": 0.5, "Mix": 0.4, "Scene": 0.1})
    analyzer._fbclf = Clf("tags+frames_v1+video_v1+frames_v3", {"Dance": 0.05, "Mix": 0.9, "Scene": 0.05})
    analyzer._category_classifier = lambda: analyzer._catclf
    analyzer._fallback_classifier = lambda: analyzer._fbclf

    def signals(video, fallback=False):
        calls.append(fallback)
        return {}, {"category": "Dance"}, ({"p": 0.5} if fallback else {})

    analyzer.qwen_signals = signals
    r = analyzer.run(a)
    assert calls == [False, True] and r.category == "Mix" and r.category_sure
    calls.clear()
    analyzer._catclf.probs = {"Dance": 0.9, "Mix": 0.05, "Scene": 0.05}  # 第一段就有把握 → 不加跑
    r2 = analyzer.run(b)
    assert calls == [False] and r2.category == "Dance"


def test_author_prior_category(cfg):
    """作者先驗：同作者過去幾乎都是 Dance，分類器 Dance 0.6 不夠確定 → 乘上先驗後確定；沒資料的作者不受影響。"""
    from vsort.priors import AuthorPriors
    lay = cfg.layout
    for i in range(3):
        make_video(lay.library / "mikanman" / f"v{i} [mikanman][東方Project][-][MMD][Dance].mp4")
    priors = AuthorPriors(cfg)
    assert priors.anim_type("mikanman") == ["MMD"] and priors.anim_type("nobody") is None
    dic = Dictionary(cfg.dictionary_dir)
    opts = dict(DEFAULTS) | {"category_sure_prob": 0.7}
    ai = AIResult(frames=16, category="Dance", category_sure=False, style_kind="classifier",
                  category_probs={"Dance": 0.6, "Mix": 0.3, "Scene": 0.1},
                  style={"MMD": 0.9, "Koikatsu": 0.0, "VAM": 0.0, "3D": 0.05, "2D": 0.0, "Audio": 0.0},
                  characters=[{"tag": "onozuka_komachi", "prob": 0.99, "frames": 13}])
    info, sure, reasons = decide({"info": {"title": "k", "author": "mikanman"}}, ai, dic, opts, priors=priors)
    assert sure and info["category"] == "Dance" and ai.category_prior["n"] == 3 and ai.category_probs["Dance"] > 0.7
    ai2 = AIResult(frames=16, category="Dance", category_sure=False, style_kind="classifier",
                   category_probs={"Dance": 0.6, "Mix": 0.3, "Scene": 0.1}, style={"MMD": 0.9, "3D": 0.05, "2D": 0.0},
                   characters=[{"tag": "onozuka_komachi", "prob": 0.99, "frames": 13}])
    info, sure, reasons = decide({"info": {"title": "k", "author": "someone"}}, ai2, dic, opts, priors=priors)
    assert not sure and ai2.category_prior is None and any("類別信心不足" in r for r in reasons)
    opts["author_prior_weight"] = 0
    ai3 = AIResult(frames=16, category="Dance", category_probs={"Dance": 0.6, "Mix": 0.3, "Scene": 0.1}, style_kind="classifier",
                   style={"MMD": 0.9, "3D": 0.05, "2D": 0.0}, characters=[{"tag": "onozuka_komachi", "prob": 0.99, "frames": 13}])
    info, sure, reasons = decide({"info": {"title": "k", "author": "mikanman"}}, ai3, dic, opts, priors=priors)
    assert not sure and ai3.category_prior is None


def test_audio_features_and_scaled_classifier(tmp_path):
    import numpy as np
    from vsort.audio import FEATURE_NAMES, features
    from vsort.category_train import CategoryClassifier, build_features, feature_names, fit
    assert features(None) == [0.0] * len(FEATURE_NAMES)
    sr = 16000
    t = np.arange(sr * 20) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)  # 有節拍的音
    x = features(y, sr)
    assert len(x) == len(FEATURE_NAMES) and x[FEATURE_NAMES.index("has_audio")] == 1.0
    fs = "tags+frames_v1+video_v1+audio_v1"
    row = build_features(fs, 0.5, 0.2, 0.3, {"D": 0.5, "S": 0.5}, {"category": "Mix", "scene_fraction": 0.5}, None, x)
    assert len(row) == len(feature_names(fs))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(row))) * 0.1
    X[:, 5] *= 1000  # 尺度差很多的特徵，確認有標準化
    yy = np.array([0, 1, 2] * 20)
    for c in range(3):
        X[:, c] += 20 * (yy == c)  # 前三個特徵各代表一類
    rep = fit(X, yy, 0.7, fs, 1.0, tmp_path / "c.json")
    clf = CategoryClassifier(tmp_path / "c.json")
    assert clf.mean is not None and rep["accuracy"] > 0.9
    p = clf.predict(X[1].tolist())
    assert max(p, key=p.get) == "Mix" and abs(sum(p.values()) - 1) < 1e-6


def test_rematch_unknown_author(cfg, tmp_path):
    """作者未知 的影片：畫風向量和某位作者的影片很像 → 搬到 05、作者預填；不夠像的留著。"""
    import numpy as np
    from pathlib import Path
    from vsort.analyze import UNKNOWN_AUTHOR_DIR, file_key
    from vsort.rematch import plan_rematch
    lay = cfg.layout
    emb = lay.workspace / "_cache" / "emb" / "siglip"
    emb.mkdir(parents=True)
    rng = np.random.default_rng(0)
    base_a, base_b = rng.normal(size=1152), rng.normal(size=1152)
    for i in range(3):
        v = make_video(lay.library / "Mochi" / f"s{i} [Mochi][艦隊收藏][Shiranui][3D][Scene].mp4", size=4000 + i)
        np.save(emb / f"{file_key(v)}.npy", base_a + rng.normal(scale=0.05, size=1152))
    for i in range(3):
        v = make_video(lay.library / "The Studio" / f"m{i} [The Studio][-][-][MMD][Dance].mp4", size=5000 + i)
        np.save(emb / f"{file_key(v)}.npy", base_b + rng.normal(scale=0.05, size=1152))
    like_a = make_video(lay.library / UNKNOWN_AUTHOR_DIR / "x [-][-][-][3D][Scene].mp4", size=6000)
    np.save(emb / f"{file_key(like_a)}.npy", base_a + rng.normal(scale=0.05, size=1152))
    sc.write(like_a, {"info": {"title": "x", "author": "", "anim_type": ["3D"], "category": "Scene"}, "batch": "batch-1"})
    nobody = make_video(lay.library / UNKNOWN_AUTHOR_DIR / "y [-][-][-][3D][Scene].mp4", size=7000)
    np.save(emb / f"{file_key(nobody)}.npy", rng.normal(size=1152))
    plan, n = plan_rematch(cfg)
    assert n == 2 and len(plan.ops) == 1
    op = plan.ops[0]
    assert Path(op.src) == like_a and Path(op.dst).parent == lay.review
    assert op.sidecar["info"]["author"] == "Mochi" and "畫風指紋" in op.sidecar["reasons"][0]
    assert Path(op.dst).name.startswith("x [Mochi]") and op.sidecar["batch"] == "batch-1"
