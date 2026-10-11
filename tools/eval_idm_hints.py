"""實測 IDM 下載紀錄對應與檔名線索搜尋的命中率（只讀檔名與大小，不動影片）。

用法：
  .venv\\Scripts\\python.exe tools\\eval_idm_hints.py D:\\Videos --n 30 --seed 7 --out r.csv
  --check 20：另外拿 20 部「已知答案」（檔名或 IDM 有 ID）的影片，假裝不知道再用線索搜，算準確率
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsort.config import load_config  # noqa: E402
from vsort.hints import HintSearcher, extract_hints  # noqa: E402
from vsort.idm import IdmIndex  # noqa: E402
from vsort.lookup import candidates  # noqa: E402
from vsort.process import make_parser  # noqa: E402

EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".wmv", ".m4v", ".flv", ".ts"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config()
    parser = make_parser(cfg)
    idx = IdmIndex()
    hs = HintSearcher(cfg.layout.workspace / "_cache")
    root = Path(a.root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    rng = random.Random(a.seed)
    sample = rng.sample(files, min(a.n, len(files)))

    def row(p: Path, blind: bool = False):
        parsed = parser.parse(p.name, list(p.parent.relative_to(root).parts))
        base = candidates(parsed)
        m = idx.match(p)
        idm = (m.site, m.source_id) if m and m.site else None
        hm = None
        t = time.time()
        if blind or not (base or idm):
            hm = hs.search(extract_hints(p, parsed))
        return dict(path=str(p.relative_to(root)), author=parsed.author or "",
                    parser=":".join(base[0]) if base else "",
                    idm=":".join(idm) if idm else "", idm_how=m.how if m else "",
                    idm_host=(m.host or "") if m else "",
                    hint=f"{hm.site}:{hm.source_id}" if hm else "", hint_score=hm.score if hm else "",
                    hint_reason=hm.reason if hm else "", hint_title=hm.title if hm else "",
                    secs=round(time.time() - t, 1))

    rows = []
    for i, p in enumerate(sample, 1):
        r = row(p)
        rows.append(r)
        print(f"[{i}/{len(sample)}] {r['path'][:70]} | 檔名:{r['parser']} IDM:{r['idm'] or r['idm_host']} "
              f"線索:{r['hint']} {r['hint_reason']}", flush=True)
    n = len(rows)
    base = sum(bool(r["parser"]) for r in rows)
    idm = sum(bool(r["idm"]) and not r["parser"] for r in rows)
    hint = sum(bool(r["hint"]) for r in rows)
    print(f"\n抽樣 {n} 部：檔名本來就有 ID {base}；IDM 多找到 {idm}；檔名線索再多找到 {hint}；"
          f"合計 {base + idm + hint}（{(base + idm + hint) / n:.0%}）")

    if a.check:
        known = []
        for p in rng.sample(files, len(files)):
            if len(known) >= a.check:
                break
            parsed = parser.parse(p.name, list(p.parent.relative_to(root).parts))
            if not parsed.author:
                continue
            m = idx.match(p)
            truth = candidates(parsed)[:1] or ([(m.site, m.source_id)] if m and m.site else [])
            if truth:
                known.append((p, truth[0]))
        ok = wrong = miss = 0
        for p, truth in known:
            r = row(p, blind=True)
            got = r["hint"]
            if not got:
                miss += 1
            elif got == ":".join(truth):
                ok += 1
            else:
                wrong += 1
            r["truth"] = ":".join(truth)
            rows.append({**r, "check": "ok" if got == r["truth"] else ("miss" if not got else "wrong")})
            print(f"[驗證] {r['path'][:60]} 答案 {r['truth']} 線索 {got or '-'} {r['hint_reason']}", flush=True)
        print(f"\n驗證 {len(known)} 部已知答案：猜對 {ok}、猜錯 {wrong}、沒猜 {miss}")

    if a.out:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
