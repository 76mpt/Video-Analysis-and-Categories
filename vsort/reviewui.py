"""人工確認網頁（F27）：在自己電腦開一個只有本機能連的小網頁，看預覽圖、改欄位、按確認就歸檔。

每次確認都走同一套可還原的操作紀錄；頁面上的「復原上一筆」等同 vsort undo。
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vsort import sidecar as sc
from vsort.config import Config
from vsort.journal import Run, build_undo_plan, execute, incomplete_runs, list_runs
from vsort.naming import anim_list, parse_final_name
from vsort.process import _videos
from vsort.review import confirm_plan, export

_lock = threading.Lock()


def _item(video: Path) -> dict:
    data = sc.read(video) or {}
    info = data.get("info") or parse_final_name(video.name) or {"title": video.stem}
    ai = data.get("ai") or {}
    return {
        "name": video.name,
        "info": {**{k: info.get(k) for k in ("title", "author", "works", "characters", "category")},
                 "anim_type": anim_list(info.get("anim_type"))},
        "reasons": data.get("reasons") or [],
        "suggest_authors": data.get("suggest_authors") or [],
        "style_authors": [a for a, _ in (data.get("style_authors") or [])],
        "candidates": data.get("candidates") or {},
        "original": data.get("original_name"),
        "ai": {"category": ai.get("category"), "scene_ratio": ai.get("scene_ratio"), "style": ai.get("style"),
               "tags": (ai.get("top_tags") or [])[:10]},
        "preview": sc.preview_for(video).exists(),
        "description": (data.get("description") or {}).get("text", "") or (data.get("ai") or {}).get("description", ""),
    }


class Handler(BaseHTTPRequestHandler):
    cfg: Config

    def log_message(self, *args):  # 不在命令列印每個請求
        pass

    # ---------- 工具 ----------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _video(self, name: str) -> Path | None:
        """只接受 05_Review 底下的檔名，防止讀到其他路徑。"""
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            return None
        p = self.cfg.layout.review / name
        return p if p.exists() and p.parent.resolve() == self.cfg.layout.review.resolve() else None

    # ---------- 路由 ----------

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/items":
            items = [_item(v) for v in _videos(self.cfg.layout.review, self.cfg)]
            return self._json({"items": items, "library": str(self.cfg.layout.library)})
        if url.path == "/preview":
            v = self._video(parse_qs(url.query).get("name", [""])[0])
            p = sc.preview_for(v) if v else None
            if p and p.exists():
                return self._send(200, p.read_bytes(), "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "格式錯誤"}, 400)
        url = urlparse(self.path)
        with _lock:
            if url.path == "/api/confirm":
                return self._confirm(body)
            if url.path == "/api/open":
                v = self._video(body.get("name", ""))
                if not v:
                    return self._json({"error": "找不到檔案"}, 404)
                os.startfile(v)  # 用你電腦預設的播放器開
                return self._json({"ok": True})
            if url.path == "/api/undo":
                return self._undo()
        return self._json({"error": "not found"}, 404)

    def _confirm(self, body: dict):
        if incomplete_runs(self.cfg.layout):
            return self._json({"error": "有中斷的紀錄，請先在命令列執行 vsort resume"}, 409)
        v = self._video(body.get("name", ""))
        if not v:
            return self._json({"error": "找不到檔案，可能已經歸檔"}, 404)
        plan = confirm_plan(self.cfg, [(v, body.get("info") or {})])
        if not plan.ops:
            return self._json({"error": "沒有可執行的動作"}, 400)
        batch = (sc.read(v) or {}).get("batch")
        run = execute(Run.create(self.cfg.layout, "review", plan.ops, **({"batch": batch} if batch else {})))
        export(self.cfg)
        if run.failed:
            return self._json({"error": "；".join(run.failed.values())}, 500)
        return self._json({"ok": True, "dst": plan.ops[0].dst, "run": run.run_id})

    def _undo(self):
        runs = [r for r in list_runs(self.cfg.layout) if r.kind == "review" and not r.undone_by and r.done]
        if not runs:
            return self._json({"error": "沒有可以復原的確認紀錄"}, 404)
        target = runs[-1]
        plan = build_undo_plan(target)
        run = execute(Run.create(self.cfg.layout, "undo", plan.ops, undo_of=target.run_id,
                                 partial=bool(plan.skipped)))
        export(self.cfg)
        return self._json({"ok": not run.failed, "restored": len(run.done)})


def serve(cfg: Config, port: int = 8765, open_browser: bool = True) -> None:
    cfg.layout.ensure()
    handler = type("BoundHandler", (Handler,), {"cfg": cfg})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"人工確認網頁：{url}（只有這台電腦連得到；按 Ctrl+C 結束）", flush=True)
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
<title>人工確認</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#ddddd8;--accent:#2f6fde;--warn:#b25a00;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21211f;--fg:#ecece8;--mute:#a3a39c;--line:#3a3a36;--accent:#7aa7ff;--warn:#f0a050;--ok:#7bc47f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Microsoft JhengHei","Noto Sans TC",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:16px;align-items:center;z-index:2}
header h1{font-size:18px;margin:0}header .n{color:var(--mute)}header button{margin-left:auto}
main{max-width:1200px;margin:0 auto;padding:16px 20px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;display:grid;grid-template-columns:minmax(260px,420px) 1fr;overflow:hidden}
.card img{width:100%;height:100%;object-fit:contain;background:#000;min-height:200px}
.noimg{display:flex;align-items:center;justify-content:center;color:var(--mute);background:var(--bg)}
.body{padding:14px 16px;display:grid;gap:8px}
.name{font-size:13px;color:var(--mute);word-break:break-all}
.reasons{color:var(--warn);font-size:14px}
label{display:grid;grid-template-columns:72px 1fr;gap:8px;align-items:center;font-size:14px}
input[type=text]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
.seg{display:flex;gap:6px;flex-wrap:wrap}.seg label{display:flex;grid-template-columns:none;gap:4px;border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer}
.seg input{margin:0}.ai{font-size:13px;color:var(--mute)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.chip{font-size:13px;border:1px dashed var(--line);border-radius:12px;padding:1px 8px;cursor:pointer}
.hint{color:var(--mute);font-size:13px}
.cands{display:grid;gap:4px;padding:8px;border:1px dashed var(--line);border-radius:8px}
.cand{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.chip.used{opacity:.5}
.done{opacity:.45}.msg{font-size:14px}.msg.ok{color:var(--ok)}.msg.err{color:var(--warn)}
.empty{text-align:center;color:var(--mute);padding:60px}
@media (max-width:720px){.card{grid-template-columns:1fr}}
</style></head><body>
<header><h1>人工確認</h1><span class="n" id="count"></span><button id="undo">復原上一筆</button></header>
<main id="list"></main>
<script>
const $ = (s, el=document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const join = v => Array.isArray(v) ? v.join(", ") : (v || "");
async function post(url, body){ const r = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); return [r.ok, await r.json()]; }
const pct = v => (v*100).toFixed(0)+"%";
const NAMES = {Audio: "無（聲音）"};
function cands(it){
  const c = it.candidates || {}; const rows = [];
  const row = (label, field, items) => { if(items && items.length) rows.push(`<div class="cand"><span class="hint">${label}</span>` +
    items.map(([v, p, extra]) => `<span class="chip" data-field="${field}" data-v="${esc(v)}">${esc(NAMES[v]||v)}${extra?" · "+esc(extra):""} ${pct(p)}</span>`).join(" ") + "</div>"); };
  row("類別候選", "category", Object.entries(c.category || {}));
  row("動畫類型", "anim", Object.entries(c.anim_type || {}));
  row("角色前 5", "characters", (c.characters || []).map(x => [x[0], x[2], x[1]]));
  row("作品", "works", c.works || []);
  row("畫風相近作者", "author", (c.authors || []).map(x => [x[0], x[1]]));
  return rows.length ? `<div class="cands">${rows.join("")}<div class="hint">點一下候選就會填進對應欄位</div></div>` : "";
}
function seg(name, opts, val, multi){ const sel = [].concat(val || []); return '<div class="seg">' + opts.map(o => `<label><input type="${multi?"checkbox":"radio"}" name="${name}" value="${o}" ${sel.includes(o)?"checked":""}>${o}</label>`).join("") + "</div>"; }
async function load(){
  const r = await fetch("/api/items"); const d = await r.json();
  $("#count").textContent = `${d.items.length} 部待確認`;
  const list = $("#list"); list.innerHTML = "";
  if(!d.items.length){ list.innerHTML = '<div class="empty">05_Review 沒有待確認的影片</div>'; return; }
  d.items.forEach((it, i) => {
    const f = it.info, ai = it.ai || {}, st = ai.style || {};
    const style = Object.entries(st).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k} ${(v*100).toFixed(0)}%`).join("、");
    const el = document.createElement("section"); el.className = "card";
    el.innerHTML = `
      ${it.preview ? `<img loading="lazy" src="/preview?name=${encodeURIComponent(it.name)}">` : '<div class="noimg">沒有預覽圖</div>'}
      <div class="body">
        <div class="name">${esc(it.name)}${it.original ? "<br>原始檔名：" + esc(it.original) : ""}</div>
        ${it.reasons.length ? `<div class="reasons">需要確認：${esc(it.reasons.join("；"))}</div>` : ""}
        <label>標題<input type="text" name="title" value="${esc(f.title)}"></label>
        <label>作者<span><input type="text" name="author" value="${esc(f.author)}">
          ${it.suggest_authors.map(a=>`<span class="chip" data-author="${esc(a)}">${esc(a)}</span>`).join(" ")}
          ${it.style_authors.length?`<span class="hint">畫風相近（僅供參考）：</span>`+it.style_authors.map(a=>`<span class="chip" data-author="${esc(a)}">${esc(a)}</span>`).join(" "):""}</span></label>
        ${cands(it)}
        <label>作品<input type="text" name="works" value="${esc(join(f.works))}" placeholder="多個用逗號分隔"></label>
        <label>角色<input type="text" name="characters" value="${esc(join(f.characters))}" placeholder="多個用逗號分隔"></label>
        <label>動畫類型${seg("anim"+i, ["MMD","Koikatsu","VAM","3D","2D","Audio"], f.anim_type, true)}</label>
        <label>類別${seg("cat"+i, ["Dance","Mix","Scene"], f.category)}</label>
        ${it.description ? `<div class="ai">描述模型：${esc(it.description)}</div>` : ""}
        <div class="ai">AI：${esc(ai.category || "?")}（場景畫面 ${ai.scene_ratio==null?"?":(ai.scene_ratio*100).toFixed(0)+"%"}）；畫風 ${esc(style || "?")}；標籤 ${esc((ai.tags||[]).join(", "))}</div>
        <div class="row"><button class="primary" data-act="ok">確認歸檔</button><button data-act="play">播放</button><span class="msg"></span></div>
      </div>`;
    el.querySelectorAll(".chip[data-author]").forEach(c => c.onclick = () => { $('input[name=author]', el).value = c.dataset.author; });
    el.querySelectorAll(".chip[data-field]").forEach(c => c.onclick = () => {
      const f = c.dataset.field, v = c.dataset.v;
      if(f === "category"){ const r = $(`input[name=cat${i}][value="${v}"]`, el); if(r) r.checked = true; }
      else if(f === "anim"){ const b = $(`input[name=anim${i}][value="${v}"]`, el); if(b) b.checked = !b.checked; }
      else if(f === "author"){ $('input[name=author]', el).value = v; }
      else { const inp = $(`input[name=${f}]`, el); const cur = inp.value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
             if(!cur.includes(v)) { cur.push(v); inp.value = cur.join(", "); } }
      c.classList.add("used");
    });
    $('[data-act=play]', el).onclick = () => post("/api/open", {name: it.name});
    $('[data-act=ok]', el).onclick = async () => {
      const v = n => $(`input[name=${n}]`, el).value;
      const r = n => ($(`input[name=${n}${i}]:checked`, el) || {}).value || "";
      const anim = [...el.querySelectorAll(`input[name=anim${i}]:checked`)].map(x => x.value);
      const info = {title: v("title"), author: v("author"), works: v("works"), characters: v("characters"), anim_type: anim, category: r("cat")};
      const msg = $(".msg", el);
      const [ok, res] = await post("/api/confirm", {name: it.name, info});
      if(ok){ el.classList.add("done"); msg.className = "msg ok"; msg.textContent = "已歸檔：" + res.dst; el.querySelectorAll("button,input").forEach(x => x.disabled = true); }
      else { msg.className = "msg err"; msg.textContent = res.error || "失敗"; }
    };
    list.appendChild(el);
  });
}
$("#undo").onclick = async () => { const [ok, res] = await post("/api/undo", {}); alert(ok ? `已復原 ${res.restored} 部` : (res.error || "失敗")); load(); };
load();
</script></body></html>
"""
