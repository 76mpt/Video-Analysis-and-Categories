"""動畫類型樣本標註（給自訓小分類器用）。

從你指定的資料夾平均抽樣影片（不搬檔案），開一個本機網頁讓你勾 MMD／Koikatsu／VAM／3D／2D／無（可複選），
結果存到 dict\\style_labels.csv（可以用 Excel 打開修改）。檔名或 iwara 標籤已經寫明類型的會先幫你勾好。
"""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vsort.analyze import file_key
from vsort.config import Config
from vsort.frames import contact_sheet, extract_cached
from vsort.naming import ANIM_DISPLAY, ANIM_ORDER, anim_list
from vsort.parser import Parser

HEADER = ["檔案路徑", "key", *ANIM_ORDER, "看不出來", "類別", "預填來源", "時間"]
CATEGORIES = ["Dance", "Mix", "Scene"]
_lock = threading.Lock()


def labels_path(cfg: Config) -> Path:
    return cfg.dictionary_dir / "style_labels.csv"


def load_labels(cfg: Config) -> dict[str, dict]:
    path = labels_path(cfg)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        old_format = "Audio" not in (reader.fieldnames or [])  # 舊版只有 MMD／3D／2D 三欄
        for r in reader:
            if r.get("key"):
                labels = [a for a in ANIM_ORDER if r.get(a) == "1"]
                skip = r.get("看不出來") == "1"
                if old_format and not labels and not skip:
                    labels = ["Audio"]  # 舊版什麼都沒勾＝你說的「（無）」：聲音作品附贈的影片
                out[r["key"]] = {"path": r["檔案路徑"], "labels": labels, "skip": skip,
                                 "category": r.get("類別") or "", "source": r.get("預填來源", ""),
                                 "t": r.get("時間", "")}
    return out


def save_labels(cfg: Config, labels: dict[str, dict]) -> None:
    path = labels_path(cfg)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for key, v in labels.items():
            w.writerow([v["path"], key, *("1" if a in v["labels"] else "0" for a in ANIM_ORDER),
                        "1" if v.get("skip") else "0", v.get("category", ""), v.get("source", ""), v.get("t", "")])
    tmp.replace(path)


def sample_videos(roots: list[Path], cfg: Config, n: int, per_folder: int, seed: int,
                  exclude: set[str]) -> list[Path]:
    """各作者資料夾輪流抽，避免樣本集中在少數作者。"""
    groups: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for dirpath, _, names in os.walk(root):
            for name in names:
                p = Path(dirpath) / name
                if p.suffix.lower() in cfg.video_extensions and not cfg.skipped(p, root):
                    rel = p.relative_to(root)
                    groups[rel.parts[0] if len(rel.parts) > 1 else root.name].append(p)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out: list[Path] = []
    taken: dict[str, int] = defaultdict(int)
    while len(out) < n and any(groups.values()):
        progressed = False
        for g, items in groups.items():
            while items and len(out) < n and taken[g] < per_folder:
                p = items.pop()
                if str(p) in exclude:
                    continue
                out.append(p)
                taken[g] += 1
                progressed = True
                break
        if not progressed:
            break
    return out


def prefill(path: Path, root: Path, parser: Parser, lookup=None) -> tuple[list[str], str]:
    """檔名、資料夾或 iwara 標籤寫明的類型，先幫你勾。"""
    rel = path.relative_to(root) if root in path.parents else Path(path.name)
    p = parser.parse(path.name, list(rel.parts[:-1]))
    labels, sources = list(p.anim_type), ["檔名"] if p.anim_type else []
    if lookup is not None and p.site == "iwara" and p.source_id:
        from vsort.lookup import ANIM_TAGS
        from vsort.dictionary import normalize
        try:
            info = lookup.fetch("iwara", p.source_id)
            tags = [ANIM_TAGS[normalize(t)] for t in info.tags if normalize(t) in ANIM_TAGS]
            if tags:
                labels += tags
                sources.append("iwara 標籤")
        except Exception:
            pass
    return anim_list(labels), "、".join(sources)


class LabelSession:
    def __init__(self, cfg: Config, queue: list[dict], frames: int = 16):
        self.cfg = cfg
        self.queue = queue
        self.frames = frames
        self.cache = cfg.layout.workspace / "_cache"
        self.labels = load_labels(cfg)
        self.ready: set[str] = set()
        threading.Thread(target=self._prepare, daemon=True).start()

    def _sheet_path(self, key: str) -> Path:
        return self.cache / "frames" / key / "sheet.jpg"

    def _prepare(self) -> None:
        """背景先把每部影片的關鍵幀和預覽圖做好。"""
        for item in self.queue:
            sheet = self._sheet_path(item["key"])
            if not sheet.exists():
                try:
                    frames = extract_cached(Path(item["path"]), self.frames, sheet.parent)
                    if frames:
                        contact_sheet(frames).save(sheet, quality=82)
                except Exception:
                    continue
            self.ready.add(item["key"])

    def items(self) -> list[dict]:
        out = []
        for it in self.queue:
            lab = self.labels.get(it["key"])
            out.append({**it, "done": lab is not None,
                        "labels": lab["labels"] if lab else it["prefill"],
                        "skip": bool(lab and lab.get("skip")),
                        "category": (lab or {}).get("category", ""),
                        "ready": it["key"] in self.ready or self._sheet_path(it["key"]).exists()})
        return out

    def set_label(self, key: str, labels: list[str], skip: bool, category: str = "") -> None:
        item = next((i for i in self.queue if i["key"] == key), None)
        if item is None:
            raise KeyError(key)
        with _lock:
            self.labels[key] = {"path": item["path"], "labels": anim_list(labels), "skip": skip,
                                "category": category if category in CATEGORIES else "",
                                "source": item.get("source", ""),
                                "t": datetime.now().isoformat(timespec="seconds")}
            save_labels(self.cfg, self.labels)


def pick_uncertain(cfg: Config, pool: list[Path], n: int, frames: int = 16, progress=print) -> list[Path]:
    """主動學習：用目前訓練好的分類器跑一遍候選影片，挑它最沒把握的（也偏好 3D、2D 這類樣本少的）。"""
    from vsort.style_train import CLASSIFIER_PATH, TrainedStyleClassifier
    if not CLASSIFIER_PATH.exists():
        return pool[:n]
    cache = cfg.layout.workspace / "_cache"
    clf = TrainedStyleClassifier(cache)
    scored = []
    for i, p in enumerate(pool, 1):
        try:
            key = file_key(p)
            imgs = [im for _, im in extract_cached(p, frames, cache / "frames" / key)]
            if not imgs:
                continue
            probs = clf.predict_video(key, imgs)
        except Exception:
            continue
        unsure = max(0.5 - abs(v - 0.5) for v in probs.values())
        rare = 0.15 if probs.get("3D", 0) > 0.25 or probs.get("2D", 0) > 0.25 else 0.0
        scored.append((unsure + rare, p))
        if i % 25 == 0:
            progress(f"  已評估 {i}／{len(pool)} 部候選")
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:n]]


def pick_category_uncertain(cfg: Config, pool: list[Path], n: int, progress=print, analyzer=None) -> list[Path]:
    """類別的主動學習：跑完整的 AI（畫面標籤＋Qwen 逐張＋影片模式），挑「最可能是 Mix」或類別最沒把握的。
    結果會快取，之後真的分析這些影片時不用重算。"""
    from vsort.analyze import Analyzer, options
    from vsort.priors import AuthorPriors
    from vsort.process import make_parser
    opts = options(cfg) | {"category_model": "combined"}
    an = analyzer or Analyzer(cfg, opts)
    priors = AuthorPriors(cfg, weight=float(opts.get("author_prior_weight") or 0)) if opts.get("author_prior_weight") else None
    parser = make_parser(cfg)
    scored = []
    for i, p in enumerate(pool, 1):
        try:
            r = an.run(p)
        except Exception:
            continue
        probs = r.category_probs or ({r.category: 1.0} if r.category else {})
        if not probs:
            continue
        if priors is not None and r.category_probs:  # 和實際判定一樣先套作者先驗，挑的才是真正還沒把握的
            try:
                probs, _ = priors.apply_category(probs, parser.parse(p.name, list(p.parent.parts[1:])).author)
            except Exception:
                pass
        unsure = 1.0 - max(probs.values())
        scored.append((probs.get("Mix", 0.0) + unsure, p))
        if i % 25 == 0:
            progress(f"  已評估 {i}／{len(pool)} 部候選")
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:n]]


def build_queue(cfg: Config, roots: list[Path], n: int, per_folder: int = 8, seed: int = 1,
                use_lookup: bool = True, uncertain: bool = False, progress=print,
                include_labeled: list[str] | None = None, category_uncertain: bool = False) -> list[dict]:
    from vsort.process import make_parser
    qpath = cfg.layout.workspace / "_cache" / "label_queue.json"
    existing = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else []
    done = load_labels(cfg)
    exclude = {it["path"] for it in existing} | {v["path"] for v in done.values()}
    parser = make_parser(cfg)
    lookup = None
    if use_lookup:
        from vsort.lookup import Lookup
        lookup = Lookup(cfg)
    queue = [it for it in existing if Path(it["path"]).exists()]
    if category_uncertain:  # 類別：挑最可能是 Mix、或類別最沒把握的
        queue = [it for it in queue if it["key"] not in done]
        pool = sample_videos(roots, cfg, n * 3 // 2, per_folder * 2, seed + 200, exclude)
        progress(f"從 {len(pool)} 部候選中挑最可能是 Mix 或類別最沒把握的 {n} 部（每部要跑一次 AI，會花一些時間）…")
        picks = pick_category_uncertain(cfg, pool, n, progress=progress)
    elif uncertain:  # 新一輪：只留還沒標的，加上分類器最沒把握的
        queue = [it for it in queue if it["key"] not in done]
        pool = sample_videos(roots, cfg, n * 3, per_folder * 2, seed + 100, exclude)
        progress(f"從 {len(pool)} 部候選中挑分類器最沒把握的 {n} 部（要先擷取畫面，會花一些時間）…")
        picks = pick_uncertain(cfg, pool, n, progress=progress)
    else:
        picks = sample_videos(roots, cfg, n, per_folder, seed, exclude)
    for p in picks:
        root = next((r for r in roots if r in p.parents), p.parent)
        labels, source = prefill(p, root, parser, lookup)
        queue.append({"key": file_key(p), "path": str(p), "name": p.name,
                      "folder": str(p.parent.relative_to(root)) if root in p.parents else "",
                      "prefill": labels, "source": source})
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(json.dumps(queue, ensure_ascii=False, indent=1), encoding="utf-8")
    if include_labeled:  # 把已標過某些類型的影片也放進來，方便改標（不寫進佇列檔）
        keys = {it["key"] for it in queue}
        for key, v in done.items():
            if key in keys or not set(v["labels"]) & set(include_labeled):
                continue
            p = Path(v["path"])
            root = next((r for r in roots if r in p.parents), p.parent)
            labels, source = prefill(p, root, parser, lookup) if p.exists() else ([], "")
            queue.append({"key": key, "path": str(p), "name": p.name,
                          "folder": str(p.parent.relative_to(root)) if root in p.parents else "",
                          "prefill": labels, "source": source})
    return queue


class Handler(BaseHTTPRequestHandler):
    session: LabelSession

    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            opts = json.dumps([[a, ANIM_DISPLAY.get(a, a)] for a in ANIM_ORDER], ensure_ascii=False)
            return self._send(200, PAGE.replace("__OPTS__", opts).encode(), "text/html; charset=utf-8")
        if url.path == "/api/items":
            return self._json({"items": self.session.items()})
        if url.path == "/sheet":
            key = parse_qs(url.query).get("key", [""])[0]
            if key.isalnum():
                p = self.session._sheet_path(key)
                if p.exists():
                    return self._send(200, p.read_bytes(), "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        url = urlparse(self.path)
        if url.path == "/api/label":
            try:
                self.session.set_label(body["key"], body.get("labels") or [], bool(body.get("skip")),
                                       body.get("category") or "")
            except KeyError:
                return self._json({"error": "找不到這部影片"}, 404)
            return self._json({"ok": True})
        if url.path == "/api/open":
            item = next((i for i in self.session.queue if i["key"] == body.get("key")), None)
            if item and Path(item["path"]).exists():
                os.startfile(item["path"])
                return self._json({"ok": True})
            return self._json({"error": "找不到檔案"}, 404)
        return self._json({"error": "not found"}, 404)


def serve(cfg: Config, queue: list[dict], port: int = 8767, open_browser: bool = True) -> None:
    session = LabelSession(cfg, queue)
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"標註網頁：{url}（只有這台電腦連得到；標註會即時存到 {labels_path(cfg)}；按 Ctrl+C 結束）", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>動畫類型標註</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#ddddd8;--accent:#2f6fde;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21211f;--fg:#ecece8;--mute:#a3a39c;--line:#3a3a36;--accent:#7aa7ff;--ok:#7bc47f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Microsoft JhengHei","Noto Sans TC",system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0}.hint{color:var(--mute);font-size:13px}
.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
main{max-width:1100px;margin:0 auto;padding:16px 20px;display:grid;gap:14px}
.card{background:var(--card);border:2px solid var(--line);border-radius:10px;display:grid;grid-template-columns:minmax(240px,520px) 1fr;overflow:hidden}
.card.cur{border-color:var(--accent)}.card.done{opacity:.6}
.card img{width:100%;display:block;background:#000}.wait{display:flex;align-items:center;justify-content:center;min-height:180px;color:var(--mute)}
.body{padding:12px 16px;display:grid;gap:10px;align-content:start}
.name{font-size:13px;color:var(--mute);word-break:break-all}
.opts{display:flex;gap:8px;flex-wrap:wrap}
.opt{border:1px solid var(--line);border-radius:8px;padding:6px 14px;cursor:pointer;user-select:none}
.opt.on{background:var(--accent);border-color:var(--accent);color:#fff}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.src{font-size:13px;color:var(--mute)}.st{font-size:13px;color:var(--ok)}
@media (max-width:720px){.card{grid-template-columns:1fr}}
</style></head><body>
<header><h1>動畫類型標註</h1><span id="prog"></span><div class="bar"><i id="barfill"></i></div>
<select id="filter"><option value="all">全部</option><option value="todo">只看未標</option><option value="3d">只看標成 3D 的</option></select>
<span class="hint">鍵盤：數字鍵勾動畫類型（可複選）、D／M／S 選類別，Enter＝存好跳下一部，0＝看不出來，P＝播放</span></header>
<main id="list"></main>
<script>
const $=(s,el=document)=>el.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const OPTS=__OPTS__.map(o=>o[0]), NAMES=Object.fromEntries(__OPTS__); let items=[], cur=0, filter="all";
async function post(u,b){try{const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
  if(!r.ok) throw new Error("HTTP "+r.status); return await r.json();}
  catch(e){alert("儲存失敗，標註網頁的程式可能已經關閉。請重新執行 vsort.bat label 後再標一次。（"+e.message+"）"); throw e;}}
function progress(){const d=items.filter(i=>i.done).length;$("#prog").textContent=`已標 ${d}／${items.length}`;$("#barfill").style.width=(items.length?d/items.length*100:0)+"%";}
function render(){
  const list=$("#list"); list.innerHTML="";
  items.forEach((it,i)=>{
    const el=document.createElement("section"); el.className="card"+(it.done?" done":"")+(i===cur?" cur":""); el.id="c"+i;
    el.innerHTML=`${it.ready?`<img loading="lazy" src="/sheet?key=${it.key}">`:'<div class="wait">擷取畫面中…</div>'}
      <div class="body"><div class="name">${esc(it.folder)}<br>${esc(it.name)}</div>
      <div class="opts cats">${["Dance","Mix","Scene"].map(c=>`<span class="opt cat ${it.category===c?"on":""}" data-c="${c}">${c[0]}＝${c}</span>`).join("")}<span class="src">類別（D／M／S，選填）</span></div>
      <div class="opts">${OPTS.map((o,k)=>`<span class="opt ${it.labels.includes(o)&&!it.skip?"on":""}" data-o="${o}">${k+1} ${esc(NAMES[o])}</span>`).join("")}</div>
      ${it.source?`<div class="src">${it.done?"提示（"+esc(it.source)+"）":"已依"+esc(it.source)+"預先勾選，請確認"}：${esc((it.prefill||[]).map(x=>NAMES[x]).join(", "))}</div>`:""}
      <div><button class="primary" data-a="save">存好，下一部</button> <button data-a="skip">看不出來</button> <button data-a="play">播放</button></div>
      <div class="st">${it.done?(it.skip?"已標：看不出來":"已標："+(it.labels.map(x=>NAMES[x]).join(", ")||"（未勾）")):""}</div></div>`;
    if(filter==="todo"&&it.done) el.style.display="none";
    if(filter==="3d"&&!(it.done&&it.labels.includes("3D"))) el.style.display="none";
    el.querySelectorAll(".opt[data-o]").forEach(o=>o.onclick=()=>{toggle(i,o.dataset.o)});
    el.querySelectorAll(".opt[data-c]").forEach(o=>o.onclick=()=>{setCat(i,o.dataset.c)});
    $("[data-a=save]",el).onclick=()=>save(i,false);
    $("[data-a=skip]",el).onclick=()=>save(i,true);
    $("[data-a=play]",el).onclick=()=>post("/api/open",{key:it.key});
    el.onclick=e=>{if(e.target===el||e.target.tagName==="IMG"){cur=i;mark();}};
    list.appendChild(el);
  });
  progress();
}
function setCat(i,c){const it=items[i]; it.category=it.category===c?"":c;
  $("#c"+i).querySelectorAll(".opt[data-c]").forEach(e=>e.classList.toggle("on",e.dataset.c===it.category));}
function mark(){document.querySelectorAll(".card").forEach((c,i)=>c.classList.toggle("cur",i===cur));}
function toggle(i,o){const it=items[i]; it.skip=false; it.labels=it.labels.includes(o)?it.labels.filter(x=>x!==o):[...it.labels,o];
  $("#c"+i).querySelectorAll(".opt[data-o]").forEach(e=>e.classList.toggle("on",it.labels.includes(e.dataset.o)));}
async function save(i,skip){const it=items[i]; await post("/api/label",{key:it.key,labels:skip?[]:it.labels,skip,category:it.category||""});
  it.done=true; it.skip=skip; const el=$("#c"+i); el.classList.add("done");
  $(".st",el).textContent=skip?"已標：看不出來":"已標："+(it.labels.map(x=>NAMES[x]).join(", ")||"（未勾）"); progress();
  const visible=j=>$("#c"+j).style.display!=="none";
  const nxt=items.findIndex((x,j)=>j>i&&visible(j)&&(filter!=="todo"||!x.done)); if(nxt>=0){cur=nxt;mark();$("#c"+nxt).scrollIntoView({behavior:"smooth",block:"center"});}}
document.addEventListener("keydown",e=>{if(!items.length)return;
  if(e.key>="1"&&e.key<=String(OPTS.length))toggle(cur,OPTS[+e.key-1]);
  else if(["d","m","s"].includes(e.key.toLowerCase()))setCat(cur,{d:"Dance",m:"Mix",s:"Scene"}[e.key.toLowerCase()]);
  else if(e.key==="Enter")save(cur,false); else if(e.key==="0")save(cur,true);
  else if(e.key.toLowerCase()==="p")post("/api/open",{key:items[cur].key});});
async function load(){const d=await (await fetch("/api/items")).json(); const first=!items.length; items=d.items;
  if(first){cur=Math.max(0,items.findIndex(i=>!i.done));} render(); if(first&&items[cur]) $("#c"+cur).scrollIntoView({block:"center"});}
$("#filter").onchange=e=>{filter=e.target.value;render();};
load(); setInterval(async()=>{const d=await (await fetch("/api/items")).json(); let ch=false;
  d.items.forEach((x,i)=>{if(items[i]&&x.ready&&!items[i].ready){items[i].ready=true;ch=true;}}); if(ch)render();},4000);
</script></body></html>
"""
