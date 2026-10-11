"""實測 Qwen3-VL「影片模式」判類別，並和其他做法合併比較（用你標的類別當答案）。

用法：.venv\\Scripts\\python.exe tools\\eval_qwen_video.py --out qwen_video_eval.json
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
from vsort.describe import QwenDescriber  # noqa: E402
from vsort.frames import extract_cached  # noqa: E402
from vsort.labeling import load_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qwen_video_eval.json")
    args = ap.parse_args()
    cfg = load_config()
    cache = cfg.layout.workspace / "_cache"
    items = [v for v in load_labels(cfg).values() if v.get("category") in CATS and Path(v["path"]).exists()]
    q, secs = None, []
    X, y, vid = [], [], []
    for v in items:
        p = Path(v["path"])
        k = file_key(p)
        cpath = cache / "ai" / f"{k}_qwenvideo_v1.json"
        if cpath.exists():
            r = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            q = q or QwenDescriber()
            imgs = [im for _, im in extract_cached(p, 16, cache / "frames" / k)]
            t0 = time.perf_counter()
            r = q.classify_video(imgs)
            secs.append(time.perf_counter() - t0)
            cpath.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        ai = json.loads((cache / "ai" / f"{k}.json").read_text(encoding="utf-8"))
        old = json.loads((cache / "ai" / f"{k}_describe_qwen3vl_v2.json").read_text(encoding="utf-8")).get("json")
        v1 = json.loads((cache / "ai" / f"{k}_qwenframes_v1.json").read_text(encoding="utf-8"))["stats"]
        vf = r.get("scene_fraction")
        vf = 0.5 if vf is None else float(vf)
        X.append([*features(ai["scene_ratio"], ai["dance_ratio"], ai["explicit_ratio"], old)[:3],
                  *(v1.get(x, 0) for x in ("D", "S", "O", "switches")),
                  *(1.0 if r.get("category") == c else 0.0 for c in CATS), vf])
        y.append(CATS.index(v["category"]))
        vid.append(CATS.index(r["category"]) if r.get("category") in CATS else 2)
    X, y, vid = np.array(X), np.array(y), np.array(vid)

    def recall(pred):
        return {c: round(float((pred[y == i] == i).mean()), 3) for i, c in enumerate(CATS)}

    report = {"n": len(y), "sec_per_video": round(float(np.mean(secs)), 2) if secs else None,
              "video_mode_alone": {"accuracy": round(float((vid == y).mean()), 3), "recall": recall(vid)}}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, cols in [("畫面標籤＋逐張", list(range(7))), ("畫面標籤＋逐張＋影片模式", list(range(11))),
                       ("畫面標籤＋影片模式", [0, 1, 2, 7, 8, 9, 10])]:
        for cw in (None, "balanced"):
            best = None
            for C in (0.3, 1.0, 3.0):
                p = cross_val_predict(LogisticRegression(C=C, max_iter=3000, class_weight=cw), X[:, cols], y, cv=cv,
                                      method="predict_proba")
                pr = p.argmax(1)
                conf = p.max(1) >= 0.8
                e = {"accuracy": round(float((pr == y).mean()), 3), "recall": recall(pr), "C": C,
                     "sure_share": round(float(conf.mean()), 3),
                     "accuracy_when_sure": round(float((pr[conf] == y[conf]).mean()), 3) if conf.any() else None}
                score = e["accuracy"] if cw is None else np.mean(list(e["recall"].values()))
                if best is None or score > best[0]:
                    best = (score, e)
            report[f"{name}（{'不加權' if cw is None else '平衡類別'}）"] = best[1]
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
