"""量類別（Dance／Mix／Scene）的正確率：用你在標註網頁選的類別當答案，
比較 Qwen3-VL、畫面標籤比例、以及兩者一致時的正確率。

用法：.venv\\Scripts\\python.exe tools\\eval_category.py --out category_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vsort.analyze import Analyzer, options  # noqa: E402
from vsort.config import load_config  # noqa: E402
from vsort.labeling import load_labels  # noqa: E402

CATS = ["Dance", "Mix", "Scene"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="category_eval.json")
    args = ap.parse_args()
    cfg = load_config()
    opts = options(cfg) | {"category_model": "qwen3vl"}
    an = Analyzer(cfg, opts)
    items = [v for v in load_labels(cfg).values() if v.get("category") in CATS and Path(v["path"]).exists()]
    print(f"有類別答案的影片 {len(items)} 部", flush=True)
    rows = []
    for i, v in enumerate(items, 1):
        r = an.run(Path(v["path"]))
        rows.append({"file": v["path"], "answer": v["category"], "qwen": r.category_vlm, "tags": r.category_tags,
                     "scene_ratio": r.scene_ratio})
        if i % 20 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    def acc(key):
        ok = [r for r in rows if r[key]]
        return round(sum(r[key] == r["answer"] for r in ok) / max(len(ok), 1), 3)

    agree = [r for r in rows if r["qwen"] and r["qwen"] == r["tags"]]
    report = {
        "n": len(rows),
        "answers": dict(Counter(r["answer"] for r in rows)),
        "qwen_accuracy": acc("qwen"),
        "tags_accuracy": acc("tags"),
        "agree_rate": round(len(agree) / max(len(rows), 1), 3),
        "accuracy_when_agree": round(sum(r["qwen"] == r["answer"] for r in agree) / max(len(agree), 1), 3),
        "qwen_confusion": {f"{a}->{p}": n for (a, p), n in Counter((r["answer"], r["qwen"]) for r in rows).items()},
        "tags_confusion": {f"{a}->{p}": n for (a, p), n in Counter((r["answer"], r["tags"]) for r in rows).items()},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
