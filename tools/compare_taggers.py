"""比較角色辨識模型（EVA02／PixAI／Camie）。

拿檔名裡已經寫明角色的真實影片當答案（只讀影片，不搬檔案），看各模型認不認得出那個角色。
用法：.venv\\Scripts\\python.exe tools\\compare_taggers.py D:\\Videos --n 40 --out 結果.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vsort.analyze import danbooru_to_names  # noqa: E402
from vsort.config import load_config  # noqa: E402
from vsort.dictionary import Dictionary  # noqa: E402
from vsort.frames import extract_cached  # noqa: E402
from vsort.analyze import file_key  # noqa: E402
from vsort.parser import Parser  # noqa: E402

MODELS = ["SmilingWolf/wd-eva02-large-tagger-v3", "pixai-labs/pixai-tagger-v1.0", "Camais03/camie-tagger-v2"]
THRESHOLDS = [0.3, 0.5, 0.7, 0.85]


def pick_samples(root: Path, parser: Parser, n: int, seed: int) -> list[tuple[Path, list[str]]]:
    exts = {".mp4", ".mkv", ".webm", ".mov"}
    by_work: dict[str, list] = defaultdict(list)
    for f in root.rglob("*"):
        if f.suffix.lower() not in exts:
            continue
        rel = f.relative_to(root)
        p = parser.parse(f.name, list(rel.parts[:-1]))
        if p.characters:
            by_work[p.works[0] if p.works else "?"].append((f, p.characters))
    rng = random.Random(seed)
    for v in by_work.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by_work.values()):  # 各作品輪流取，避免全是同一部
        for w in list(by_work):
            if by_work[w] and len(out) < n:
                out.append(by_work[w].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="compare_taggers.json")
    ap.add_argument("--models", default=",".join(MODELS), help="逗號分隔")
    args = ap.parse_args()

    cfg = load_config()
    dic = Dictionary(cfg.dictionary_dir)
    parser = Parser(dic)
    samples = pick_samples(Path(args.root), parser, args.n, args.seed)
    print(f"樣本 {len(samples)} 部", flush=True)
    cache = cfg.layout.workspace / "_cache" / "frames"
    frames = {}
    for f, _ in samples:
        frames[str(f)] = [im for _, im in extract_cached(f, args.frames, cache / file_key(f))]
    print("關鍵幀擷取完成", flush=True)

    from vsort.tagger import make_tagger
    report = {"samples": [{"file": str(f), "answer": chars} for f, chars in samples], "models": {}}
    old = json.loads(Path(args.out).read_text(encoding="utf-8")) if Path(args.out).exists() else {}
    report["models"].update(old.get("models", {}))  # 保留之前跑過的模型結果
    for name in args.models.split(","):
        t0 = time.time()
        tagger = make_tagger(name)
        load = time.time() - t0
        chars = tagger.character_set()
        name_cache: dict[str, str] = {}
        per_video = []
        t1 = time.time()
        for f, answer in samples:
            preds = tagger.predict(frames[str(f)])
            best: dict[str, float] = {}
            for fr in preds:
                for tag, p in fr.items():
                    if tag in chars and p >= THRESHOLDS[0]:  # 低於最低門檻的不用算，省下大量字典比對
                        if tag not in name_cache:
                            name_cache[tag] = danbooru_to_names(tag, dic)[0]
                        mapped = name_cache[tag]
                        best[mapped] = max(best.get(mapped, 0), p)
            per_video.append({"answer": answer, "scores": {k: round(v, 3) for k, v in
                                                           sorted(best.items(), key=lambda kv: -kv[1])[:6]}})
        infer = (time.time() - t1) / max(len(samples), 1)
        stats = {}
        for thr in THRESHOLDS:
            hit = sum(1 for v in per_video if any(v["scores"].get(a, 0) >= thr for a in v["answer"]))
            extra = sum(len([k for k, s in v["scores"].items() if s >= thr and k not in v["answer"]]) for v in per_video)
            stats[str(thr)] = {"recall": round(hit / len(per_video), 3), "extra_per_video": round(extra / len(per_video), 2)}
        report["models"][name] = {"load_s": round(load, 1), "sec_per_video": round(infer, 2), "stats": stats,
                                  "per_video": per_video}
        print(name, json.dumps({"sec_per_video": round(infer, 2), **stats}, ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")  # 每跑完一個就存
        del tagger
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已輸出", args.out)


if __name__ == "__main__":
    main()
