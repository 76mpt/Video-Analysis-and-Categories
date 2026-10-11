"""完整流程實測：拿真實影片跑「檔名解析 → 網路查詢 → AI 判定」（不搬任何檔案），
統計自動歸檔（已確定）的比例、送人工確認的原因、各步驟耗時，以及能對答案的部分的正確率。

用法：.venv\\Scripts\\python.exe tools\\benchmark.py D:\\Videos --n 30 --out benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vsort.analyze import Analyzer, decide, options  # noqa: E402
from vsort.config import load_config  # noqa: E402
from vsort.dictionary import Dictionary  # noqa: E402
from vsort.labeling import load_labels  # noqa: E402
from vsort.lookup import Lookup, make_enricher  # noqa: E402
from vsort.process import info_from_parsed, make_parser, merge_info  # noqa: E402


def pick(root: Path, cfg, n: int, seed: int, exclude: set[str]) -> list[Path]:
    groups = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in cfg.video_extensions and str(p) not in exclude and not cfg.skipped(p, root):
            rel = p.relative_to(root)
            groups[rel.parts[0] if len(rel.parts) > 1 else "."].append(p)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out = []
    while len(out) < n and any(groups.values()):
        for g in list(groups):
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    cfg = load_config()
    opts = options(cfg)
    root = Path(args.root)
    labels = load_labels(cfg)
    labeled = {v["path"]: v for v in labels.values()}
    videos = pick(root, cfg, args.n, args.seed, exclude=set())  # 標過的也可以抽到，用來對答案
    parser = make_parser(cfg)
    dic = Dictionary(cfg.dictionary_dir)
    enrich = make_enricher(Lookup(cfg), dic)
    analyzer = Analyzer(cfg, opts)
    from vsort.priors import AuthorPriors
    priors = AuthorPriors(cfg)
    t_load = time.perf_counter()
    analyzer.tagger, analyzer.style  # 先載入模型，耗時另外算
    if opts.get("category_model") == "qwen3vl":
        analyzer._get_describer("qwen3vl")
    load_s = time.perf_counter() - t_load
    print(f"模型載入 {load_s:.0f} 秒；樣本 {len(videos)} 部", flush=True)

    rows = []
    for i, v in enumerate(videos, 1):
        rel = v.relative_to(root)
        t0 = time.perf_counter()
        p = parser.parse(v.name, list(rel.parts[:-1]))
        info = info_from_parsed(p)
        data = {"parsed": p.to_dict()}
        r = enrich(p, data, path=v)
        if r:
            info = merge_info(info, r.info)
            data.update(r.data)
        t_lookup = time.perf_counter() - t0
        ai = analyzer.run(v)
        data["info"] = info
        final, sure, reasons = decide(data, ai, dic, opts, priors=priors)
        lab = labeled.get(str(v))
        rows.append({
            "file": str(v), "sure": sure, "reasons": reasons, "info": final,
            "filename_characters": p.characters, "ai_category_tags": ai.category_tags,
            "ai_category_vlm": ai.category_vlm, "timings": {"parse_lookup": round(t_lookup, 2), **ai.timings},
            "label_anim": lab["labels"] if lab else None, "label_category": (lab or {}).get("category") or None,
        })
        print(f"  {i}/{len(videos)} {'已確定' if sure else '可能誤判'} {v.name[:50]}", flush=True)

    n = len(rows)
    stats: dict = {"n": n, "model_load_s": round(load_s, 1)}
    stats["auto_filed"] = round(sum(r["sure"] for r in rows) / n, 3)
    reason_kinds = Counter()
    for r in rows:
        for x in r["reasons"]:
            reason_kinds[x.split("（")[0].split("：")[0]] += 1
    stats["reasons"] = dict(reason_kinds.most_common())
    keys = ["parse_lookup", "frames", "tagger", "style", "category", "audio"]
    stats["avg_seconds"] = {k: round(sum(r["timings"].get(k, 0) for r in rows) / n, 2) for k in keys}
    stats["avg_total_seconds"] = round(sum(stats["avg_seconds"].values()), 2)
    with_chars = [r for r in rows if r["filename_characters"]]
    if with_chars:
        stats["character_agree_with_filename"] = round(
            sum(bool(set(dic.display_characters(r["filename_characters"])) & set(r["info"].get("characters") or []))
                for r in with_chars)
            / len(with_chars), 3)
        stats["character_samples"] = len(with_chars)
    both = [r for r in rows if r["ai_category_vlm"] and r["ai_category_tags"]]
    if both:
        stats["category_qwen_vs_tags_agree"] = round(
            sum(r["ai_category_vlm"] == r["ai_category_tags"] for r in both) / len(both), 3)
    lab_cat = [r for r in rows if r["label_category"]]
    if lab_cat:
        stats["category_correct_vs_label"] = round(
            sum(r["info"].get("category") == r["label_category"] for r in lab_cat) / len(lab_cat), 3)
        stats["category_label_samples"] = len(lab_cat)
    stats["anim_types"] = dict(Counter(",".join(r["info"].get("anim_type") or []) or "-" for r in rows))
    Path(args.out).write_text(json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
