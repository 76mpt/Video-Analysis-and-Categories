"""試跑描述模型（JoyCaption、Qwen3-VL），輸出每部影片的描述給你比較。

樣本取自標註網頁的影片（用已經擷取好的關鍵幀，不讀原檔）。
用法：.venv\\Scripts\\python.exe tools\\compare_describers.py --n 12 --out 描述比較.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vsort.config import load_config  # noqa: E402
from vsort.describe import DESCRIBERS  # noqa: E402


def _as_list(v) -> list[str]:
    if not v:
        return []
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--models", default="qwen3vl,joycaption")
    ap.add_argument("--out", default="describe_compare.csv")
    args = ap.parse_args()

    cfg = load_config()
    cache = cfg.layout.workspace / "_cache"
    queue = json.loads((cache / "label_queue.json").read_text(encoding="utf-8"))
    samples = []
    for it in queue:
        fdir = cache / "frames" / it["key"]
        meta = fdir / "frames.json"
        if meta.exists():
            n = len(json.loads(meta.read_text(encoding="utf-8"))["times"])
            samples.append((it, [Image.open(fdir / f"{i:02d}.jpg").convert("RGB") for i in range(n)]))
        if len(samples) >= args.n:
            break
    print(f"樣本 {len(samples)} 部", flush=True)

    rows = {it["key"]: {"影片": it["path"]} for it, _ in samples}
    for name in args.models.split(","):
        t0 = time.time()
        model = DESCRIBERS[name]()
        print(f"{name} 載入 {time.time() - t0:.0f} 秒", flush=True)
        t1 = time.time()
        for it, imgs in samples:
            try:
                r = model.describe(imgs)
            except Exception as e:  # 單部失敗不影響其他
                r = {"text": f"（錯誤：{e}）", "json": {}}
            rows[it["key"]][f"{name} 描述"] = r["text"]
            j = r.get("json") or {}
            rows[it["key"]][f"{name} 類別"] = j.get("category", "")
            rows[it["key"]][f"{name} 動畫類型"] = ",".join(_as_list(j.get("animation")))
            rows[it["key"]][f"{name} 角色"] = ",".join(_as_list(j.get("characters")))
        print(f"{name} 每部 {(time.time() - t1) / max(len(samples), 1):.1f} 秒", flush=True)
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    cols = ["影片"] + [c for c in next(iter(rows.values())).keys() if c != "影片"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows.values())
    print("已輸出", args.out)


if __name__ == "__main__":
    main()
