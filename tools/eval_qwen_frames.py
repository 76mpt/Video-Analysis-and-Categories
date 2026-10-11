"""實測 Qwen3-VL「逐張判斷」能不能提升類別正確率（特別是 Mix）。

用你在標註網頁選的類別當答案；結果快取在 _cache/ai/<key>_qwenframes_v<版本>.json。
用法：.venv\\Scripts\\python.exe tools\\eval_qwen_frames.py --out qwen_frames_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vsort.analyze import file_key  # noqa: E402
from vsort.category_train import CATS, features  # noqa: E402
from vsort.config import load_config  # noqa: E402
from vsort.describe import FRAMES_PROMPT_VERSION, QwenDescriber  # noqa: E402
from vsort.frames import extract_cached  # noqa: E402
from vsort.labeling import load_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qwen_frames_eval.json")
    ap.add_argument("--version", type=int, default=FRAMES_PROMPT_VERSION, help="逐張判斷提示詞版本")
    args = ap.parse_args()
    cfg = load_config()
    cache = cfg.layout.workspace / "_cache"
    items = [v for v in load_labels(cfg).values() if v.get("category") in CATS and Path(v["path"]).exists()]
    print(f"有類別答案的影片 {len(items)} 部", flush=True)
    q = None
    rows, t_total, n_run = [], 0.0, 0
    for i, v in enumerate(items, 1):
        p = Path(v["path"])
        key = file_key(p)
        cpath = cache / "ai" / f"{key}_qwenframes_v{args.version}.json"
        if cpath.exists():
            res = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            if q is None:
                q = QwenDescriber()
            imgs = [im for _, im in extract_cached(p, 16, cache / "frames" / key)]
            t0 = time.perf_counter()
            res = q.classify_frames(imgs, version=args.version)
            t_total += time.perf_counter() - t0
            n_run += 1
            cpath.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        ai = json.loads((cache / "ai" / f"{key}.json").read_text(encoding="utf-8"))
        op = cache / "ai" / f"{key}_describe_qwen3vl_v2.json"  # 舊版整部描述，新標的影片可能沒有
        old = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}
        rows.append({"file": str(p), "answer": v["category"], "frames_cat": res.get("category"),
                     "stats": res.get("stats") or {}, "ai": {k: ai[k] for k in ("scene_ratio", "dance_ratio", "explicit_ratio")},
                     "old_json": old.get("json") or {}})
        if i % 20 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    y = np.array([CATS.index(r["answer"]) for r in rows])

    def recall(pred):
        return {c: round(float((pred[y == i] == i).mean()), 3) for i, c in enumerate(CATS)}

    fc = np.array([CATS.index(r["frames_cat"]) if r["frames_cat"] in CATS else 2 for r in rows])
    old = np.array([CATS.index(r["old_json"].get("category")) if r["old_json"].get("category") in CATS else 2 for r in rows])
    report = {"n": len(rows), "sec_per_video": round(t_total / max(n_run, 1), 2),
              "old_qwen": {"accuracy": round(float((old == y).mean()), 3), "recall": recall(old)},
              "frames_rule": {"accuracy": round(float((fc == y).mean()), 3), "recall": recall(fc)}}

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    base = np.array([features(r["ai"]["scene_ratio"], r["ai"]["dance_ratio"], r["ai"]["explicit_ratio"], r["old_json"])
                     for r in rows])
    fr = np.array([[r["stats"].get(k, 0.0) for k in ("D", "S", "O", "switches", "g", "m", "p", "z")] for r in rows])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, X in [("目前的合併分類器", base), ("合併＋逐張判斷", np.concatenate([base, fr], 1)),
                    ("只用逐張判斷", fr)]:
        best = None
        for C in (0.3, 1.0, 3.0):
            for cw in (None, "balanced"):
                pr = cross_val_predict(LogisticRegression(C=C, max_iter=3000, class_weight=cw), X, y, cv=cv,
                                       method="predict_proba")
                pred = pr.argmax(1)
                score = float((pred == y).mean())
                conf = pr.max(1) >= 0.8
                rec = recall(pred)
                entry = {"accuracy": round(score, 3), "recall": rec, "C": C, "class_weight": cw,
                         "sure_share": round(float(conf.mean()), 3),
                         "accuracy_when_sure": round(float((pred[conf] == y[conf]).mean()), 3) if conf.any() else None}
                # 以「平均找回率」挑，避免為了總正確率完全放棄 Mix
                key_score = np.mean(list(rec.values()))
                if best is None or key_score > best[0]:
                    best = (key_score, entry)
        report[name] = best[1]
    Path(args.out).write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
