"""網頁首頁：在瀏覽器裡挑要分類的影片（可複選資料夾或檔案），按一下就跑完整流程，跑完接到人工確認頁。

一個本機伺服器同時提供：
- /          首頁：瀏覽電腦裡的資料夾、勾選影片或整個資料夾、開始分類、看進度與各資料夾數量
- /review    人工確認（原本的 vsort review --web）
只有這台電腦連得到（127.0.0.1）。
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vsort.config import Config
from vsort.journal import incomplete_runs
from vsort.analyze import UNKNOWN_AUTHOR_DIR
from vsort.pipeline import Job, pending_in_workspace
from vsort.pickdialog import pick_in_subprocess

_pick_lock = threading.Lock()  # 一次只開一個選擇視窗
from vsort.reviewui import PAGE as REVIEW_PAGE
from vsort.reviewui import Handler as ReviewHandler


def list_drives() -> list[str]:
    if hasattr(os, "listdrives"):
        return [d for d in os.listdrives()]
    return ["/"]


def browse(cfg: Config, path: str) -> dict:
    """列出一個資料夾底下的子資料夾與影片檔（不遞迴）。空路徑＝磁碟機清單。"""
    if not path:
        return {"path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in list_drives()], "files": []}
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(path)
    exts = cfg.video_extensions
    dirs, files = [], []
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        entries = []
    for e in entries:
        try:
            if e.is_dir():
                if e.name.startswith((".", "$")) or e.resolve() in (cfg.layout.workspace.resolve(), cfg.layout.library.resolve()):
                    continue
                dirs.append({"name": e.name, "path": str(e)})
            elif e.suffix.lower() in exts:
                files.append({"name": e.name, "path": str(e), "size": e.stat().st_size})
        except OSError:
            continue
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}


def count_videos(folder: Path, cfg: Config) -> int:
    from vsort.process import _videos
    return len(_videos(folder, cfg))


class Handler(ReviewHandler):
    job: Job

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, HOME.encode(), "text/html; charset=utf-8")
        if url.path == "/review":
            return self._send(200, REVIEW_PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/browse":
            path = parse_qs(url.query).get("path", [""])[0]
            try:
                return self._json(browse(self.cfg, path))
            except (FileNotFoundError, OSError):
                return self._json({"error": "找不到這個資料夾"}, 404)
        if url.path == "/api/batches":
            from vsort.batches import list_batches
            return self._json({"batches": list_batches(self.cfg)[:50]})
        if url.path == "/api/status":
            lay = self.cfg.layout
            stages = [(label, count_videos(p, self.cfg)) for label, p in lay.stage_dirs()]
            return self._json({"stages": stages, "job": self.job.snapshot(),
                               "paths": {"root": str(self.cfg.root_dir or ""), "workspace": str(lay.workspace),
                                         "library": str(lay.library)},
                               "incomplete": [r.run_id for r in incomplete_runs(lay)],
                               "pending_in_workspace": pending_in_workspace(self.cfg),
                               "unknown_authors": count_videos(lay.library / UNKNOWN_AUTHOR_DIR, self.cfg),
                               "default_mode": self.cfg.data["ingest"]["mode"],
                               "start_dirs": [str(x) for x in self.cfg.data.get("web", {}).get("start_dirs", [])]})
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/start":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "格式錯誤"}, 400)
            paths = [Path(p) for p in body.get("paths") or [] if p]
            paths = [p for p in paths if p.exists()]
            try:
                self.job.start(self.cfg, paths, copy=body.get("copy"))
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 400 if "勾選" in str(exc) else 409)
            return self._json({"ok": True})
        if url.path == "/api/set_root":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "格式錯誤"}, 400)
            root = Path(str(body.get("path") or ""))
            if not root.is_absolute() or not root.is_dir():
                return self._json({"error": "請給一個存在的資料夾完整路徑"}, 400)
            if self.job.busy:
                return self._json({"error": "分類執行中，等它跑完再改位置"}, 409)
            busy = pending_in_workspace(self.cfg) + count_videos(self.cfg.layout.review, self.cfg)
            if busy:
                return self._json({"error": f"工作區還有 {busy} 部影片在分類中或等人工確認，先處理完（或整批還原）再改位置"}, 409)
            if not self.cfg.path:
                return self._json({"error": "沒有設定檔可以寫入"}, 500)
            from vsort.config import load_config, set_root_dir
            set_root_dir(self.cfg.path, root)
            type(self).cfg = load_config(self.cfg.path)
            type(self).cfg.layout.ensure()
            lay = type(self).cfg.layout
            return self._json({"ok": True, "paths": {"root": str(root), "workspace": str(lay.workspace), "library": str(lay.library)}})
        if url.path == "/api/rematch":
            try:
                self.job.start_rematch(self.cfg)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            return self._json({"ok": True})
        if url.path == "/api/undo_batch":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "格式錯誤"}, 400)
            batch = str(body.get("batch") or "")
            if not batch.startswith("batch-"):
                return self._json({"error": "批次編號不對"}, 400)
            from vsort.batches import plan_undo_batch
            plans = plan_undo_batch(self.cfg, batch)
            if not plans:
                return self._json({"error": "這批已經全部還原，或找不到紀錄"}, 404)
            summary = [{"run": r.run_id, "kind": r.kind, "n": len(p.ops), "skipped": [[s, why] for s, why in p.skipped],
                        "partial": partial} for r, p, partial in plans]
            if not body.get("apply"):
                return self._json({"plans": summary})
            try:
                self.job.start_undo(self.cfg, batch)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            return self._json({"ok": True, "plans": summary})
        if url.path == "/api/answer":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "格式錯誤"}, 400)
            try:
                self.job.answer({"accept": bool(body.get("accept")), "to_review": body.get("to_review") or []})
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            return self._json({"ok": True})
        if url.path == "/api/pick":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "格式錯誤"}, 400)
            kind = body.get("kind")
            if kind not in ("files", "folder"):
                return self._json({"error": "kind 要是 files 或 folder"}, 400)
            if not _pick_lock.acquire(blocking=False):
                return self._json({"error": "選擇視窗已經開著，請先完成它"}, 409)
            try:
                paths = pick_in_subprocess(kind)
            except Exception as exc:
                return self._json({"error": f"選擇視窗開不起來：{exc}"}, 500)
            finally:
                _pick_lock.release()
            items = []
            for p in paths:
                q = Path(p)
                if q.is_dir():
                    items.append({"name": q.name or str(q), "path": str(q), "kind": "dir"})
                elif q.is_file() and q.suffix.lower() in self.cfg.video_extensions:
                    items.append({"name": q.name, "path": str(q), "kind": "file", "size": q.stat().st_size})
            return self._json({"items": items})
        if url.path == "/api/confirm" and self.job.busy:
            return self._json({"error": "分類執行中，等它跑完再確認"}, 409)
        return super().do_POST()


def serve(cfg: Config, port: int = 8765, open_browser: bool = True, page: str = "/") -> None:
    cfg.layout.ensure()
    handler = type("BoundHandler", (Handler,), {"cfg": cfg, "job": Job()})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}{page}"
    print(f"網頁：{url}（只有這台電腦連得到；首頁挑影片分類，/review 人工確認；按 Ctrl+C 結束）", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


HOME = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>影片分類</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#ddddd8;--accent:#2f6fde;--warn:#b25a00;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21211f;--fg:#ecece8;--mute:#a3a39c;--line:#3a3a36;--accent:#7aa7ff;--warn:#f0a050;--ok:#7bc47f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Microsoft JhengHei","Noto Sans TC",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:16px;align-items:center;z-index:2}
header h1{font-size:18px;margin:0}header a{color:var(--accent)}header .stages{margin-left:auto;color:var(--mute);font-size:13px}
main{max-width:1300px;margin:0 auto;padding:16px 20px;display:grid;grid-template-columns:minmax(360px,1.3fr) minmax(300px,1fr);gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:grid;gap:8px;align-content:start}
.panel h2{font-size:15px;margin:0;color:var(--mute)}
.crumb{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.crumb input{flex:1;min-width:200px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
.list{max-height:62vh;overflow:auto;border:1px solid var(--line);border-radius:8px}
.row{display:flex;gap:8px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--line)}.row:last-child{border-bottom:0}
.row input{margin:0}.row .name{flex:1;word-break:break-all}.row .name.dir{cursor:pointer;color:var(--accent)}.row .sz{color:var(--mute);font-size:12px}
.row.sel{background:color-mix(in srgb,var(--accent) 10%,transparent)}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.5;cursor:default}
.basket .row{padding:4px 8px}.basket .row .kind{font-size:12px;color:var(--mute)}
.opts{display:flex;gap:14px;flex-wrap:wrap;align-items:center}.opts label{display:flex;gap:4px;align-items:center}
.log{font:13px/1.45 Consolas,monospace;white-space:pre-wrap;max-height:40vh;overflow:auto;background:var(--bg);border-radius:8px;padding:8px}
.state{font-weight:bold}.state.running,.state.waiting{color:var(--warn)}.state.done{color:var(--ok)}.state.error{color:#c0392b}
.plan{border:2px solid var(--accent);border-radius:10px;padding:10px;display:grid;gap:8px;background:var(--card)}
.plan table{width:100%;border-collapse:collapse;font-size:13px}.plan th,.plan td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top;word-break:break-all}
.plan .lib{color:var(--ok)}.plan .rev{color:var(--warn)}.plan .tbl{max-height:45vh;overflow:auto}
.hint{color:var(--mute);font-size:13px}.empty{color:var(--mute);padding:20px;text-align:center}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>影片分類</h1><a href="/review">人工確認 →</a><span class="stages" id="stages"></span></header>
<div style="max-width:1300px;margin:8px auto 0;padding:0 20px" class="hint">分類位置：<span id="paths"></span> <button id="setroot" style="padding:2px 10px">更改…</button></div>
<main>
  <section class="panel">
    <h2>挑要分類的影片（可複選資料夾或檔案）</h2>
    <div class="crumb"><button class="primary" id="pickfiles">用視窗挑檔案…</button><button class="primary" id="pickdir">用視窗挑資料夾…</button><span class="hint">或在下面直接瀏覽勾選</span></div>
    <div class="crumb"><button id="up">↑ 上一層</button><input id="path" placeholder="貼上資料夾路徑後按 Enter，空白＝磁碟機清單"><button id="go">前往</button></div>
    <div class="list" id="list"></div>
    <div class="hint">勾資料夾＝裡面的影片全部（含子資料夾）；點資料夾名稱可以進去只挑部分檔案。工作區與收藏庫不會列出。</div>
  </section>
  <section class="panel">
    <h2>已挑選 <span id="n">0</span> 項</h2>
    <div class="basket list" id="basket"><div class="empty">還沒挑選</div></div>
    <div class="opts">
      <label><input type="radio" name="mode" value="move">搬移（原檔會移進工作區）</label>
      <label><input type="radio" name="mode" value="copy">複製（原檔留著）</label>
      <button class="primary" id="start">開始分類</button><button id="clear">清空</button>
    </div>
    <div class="opts"><button id="rematch">重新比對未知作者</button><span class="hint" id="unk"></span></div>
    <div><span class="state" id="state"></span> <span id="step"></span></div>
    <div id="plan"></div>
    <div class="log" id="log"></div>
    <div id="result"></div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>歷史批次（每一批都可以整批還原：搬回原位置、刪掉複本，含你人工確認過的）</h2>
    <div id="batches" class="hint">載入中…</div>
  </section>
</main>
<script>
const $=(s,el=document)=>el.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=b=>b>=1e9?(b/1e9).toFixed(1)+" GB":b>=1e6?(b/1e6).toFixed(0)+" MB":Math.round(b/1e3)+" KB";
let cur="", parent=null, picked=new Map(), lastState="";
async function post(u,b){const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});return [r.ok,await r.json()];}
async function browse(p){
  const r=await fetch("/api/browse?path="+encodeURIComponent(p)); const d=await r.json();
  if(d.error){alert(d.error);return;}
  cur=d.path; parent=d.parent; $("#path").value=cur; try{localStorage.setItem("vsort.dir",cur);}catch(e){}
  const list=$("#list"); list.innerHTML="";
  const rows=[...d.dirs.map(x=>({...x,kind:"dir"})),...d.files.map(x=>({...x,kind:"file"}))];
  if(!rows.length){list.innerHTML='<div class="empty">這裡沒有子資料夾或影片</div>';}
  rows.forEach(x=>{
    const el=document.createElement("div"); el.className="row"+(picked.has(x.path)?" sel":"");
    el.innerHTML=`<input type="checkbox" ${picked.has(x.path)?"checked":""}><span class="name ${x.kind}">${x.kind==="dir"?"📁 ":"🎬 "}${esc(x.name)}</span>${x.size?`<span class="sz">${fmt(x.size)}</span>`:""}`;
    $("input",el).onchange=e=>{if(e.target.checked)picked.set(x.path,x);else picked.delete(x.path);el.classList.toggle("sel",e.target.checked);renderBasket();};
    if(x.kind==="dir") $(".name",el).onclick=()=>browse(x.path);
    list.appendChild(el);
  });
}
function renderBasket(){
  const b=$("#basket"); $("#n").textContent=picked.size;
  if(!picked.size){b.innerHTML='<div class="empty">還沒挑選</div>';return;}
  b.innerHTML="";
  for(const [p,x] of picked){const el=document.createElement("div");el.className="row";
    el.innerHTML=`<span class="kind">${x.kind==="dir"?"資料夾":"檔案"}</span><span class="name">${esc(p)}</span><button>移除</button>`;
    $("button",el).onclick=()=>{picked.delete(p);renderBasket();browse(cur);}; b.appendChild(el);}
}
async function pick(kind){
  const btns=[$("#pickfiles"),$("#pickdir")]; btns.forEach(b=>b.disabled=true);
  try{const [ok,res]=await post("/api/pick",{kind}); if(!ok){alert(res.error||"失敗");return;}
    res.items.forEach(x=>picked.set(x.path,x)); renderBasket(); browse(cur);
  } finally{btns.forEach(b=>b.disabled=false);}
}
$("#pickfiles").onclick=()=>pick("files"); $("#pickdir").onclick=()=>pick("folder");
$("#up").onclick=()=>browse(parent??"");
$("#go").onclick=()=>browse($("#path").value.trim());
$("#path").onkeydown=e=>{if(e.key==="Enter")browse($("#path").value.trim());};
$("#clear").onclick=()=>{picked.clear();renderBasket();browse(cur);};
$("#rematch").onclick=async()=>{const [ok,res]=await post("/api/rematch",{}); if(!ok)alert(res.error||"失敗"); poll();};
$("#setroot").onclick=async()=>{
  const [ok,res]=await post("/api/pick",{kind:"folder"}); if(!ok){alert(res.error||"失敗");return;}
  if(!res.items.length)return; const p=res.items[0].path;
  if(!confirm(`把分類位置改成 ${p}？\n之後：${p}\\sorting\\01～05 放分類中的影片，作者資料夾直接放在 ${p} 底下。\n（原本收藏庫裡的影片不會自動搬過去）`))return;
  const [ok2,res2]=await post("/api/set_root",{path:p}); if(!ok2){alert(res2.error||"失敗");return;} poll(); loadBatches();
};
$("#start").onclick=async()=>{
  const copy=$("input[name=mode]:checked").value==="copy";
  if(!picked.size&&!confirm("沒有挑選新影片。要接著處理工作區裡還沒判定完的影片嗎？"))return;
  const [ok,res]=await post("/api/start",{paths:[...picked.keys()],copy});
  if(!ok){alert(res.error||"失敗");return;}
  picked.clear();renderBasket();poll();
};
function renderPlan(j){
  const box=$("#plan"); const p=j.pending;
  if(j.state!=="waiting"||!p){box.innerHTML="";return;}
  if(box.dataset.key===p.kind+p.rows.length)return;  // 不要每次輪詢都重畫（會弄掉你勾的）
  box.dataset.key=p.kind+p.rows.length;
  if(p.kind==="ingest"){
    box.innerHTML=`<div class="plan"><b>試跑：${esc(p.action)} ${p.rows.length} 部影片到 01_Inbox</b>，略過 ${p.skipped.length} 個${p.non_video?`（另有 ${p.non_video} 個非影片檔不處理）`:""}。原檔${p.action==="複製"?"會留著":"會被搬走（可還原）"}。
      <div class="tbl"><table><tr><th>#</th><th>原檔</th><th>放到</th></tr>${p.rows.map((r,i)=>`<tr><td>${i+1}</td><td>${esc(r.src)}</td><td>${esc(r.dst)}${r.note?`<br><span class="hint">${esc(r.note)}</span>`:""}</td></tr>`).join("")}
      ${p.skipped.map(([s,why])=>`<tr><td>略過</td><td>${esc(s)}</td><td class="hint">${esc(why)}</td></tr>`).join("")}</table></div>
      <div class="opts"><button class="primary" id="accept">確認，${esc(p.action)}進 01_Inbox 並繼續</button><button id="reject">取消（不動任何檔案）</button></div></div>`;
  } else {
    box.innerHTML=`<div class="plan"><b>AI 判定試跑：已確定 ${p.library} 部會進 04 收藏庫，可能誤判 ${p.review} 部會進 05 人工確認。</b>
      <span class="hint">覺得不對的可以勾「改送人工確認」，到人工確認頁再改欄位。</span>
      <div class="tbl"><table><tr><th>改送 05</th><th>原檔</th><th>結果</th><th>去向／原因</th></tr>${p.rows.map((r,i)=>`<tr><td>${r.to==="library"?`<input type="checkbox" data-src="${esc(r.src)}">`:"✓"}</td><td>${esc(r.name)}</td><td>${esc(r.dst_name)}</td><td class="${r.to==="library"?"lib":"rev"}">${r.to==="library"?"04 / "+esc(r.folder):"05 人工確認"}${r.reasons.length?`<br><span class="hint">${esc(r.reasons.join("；"))}</span>`:""}</td></tr>`).join("")}</table></div>
      <div class="opts"><button class="primary" id="accept">確認歸檔</button><button id="reject">取消（影片留在 03，之後可接著跑）</button></div></div>`;
  }
  $("#accept",box).onclick=async()=>{const to_review=[...box.querySelectorAll("input[data-src]:checked")].map(x=>x.dataset.src);
    const [ok,res]=await post("/api/answer",{accept:true,to_review}); if(!ok)alert(res.error||"失敗"); box.dataset.key=""; poll();};
  $("#reject",box).onclick=async()=>{const [ok,res]=await post("/api/answer",{accept:false}); if(!ok)alert(res.error||"失敗"); box.dataset.key=""; poll();};
}
const NOW={"01":"01 原始輸入","02":"02 分流","03":"03 待判定","04":"04 收藏庫","05":"05 人工確認"};
async function loadBatches(){
  const d=await (await fetch("/api/batches")).json(); const box=$("#batches");
  if(!d.batches.length){box.innerHTML='<span class="hint">還沒有任何批次</span>';return;}
  box.innerHTML=`<div class="tbl"><table><tr><th>批次</th><th>時間</th><th>來源</th><th>部數</th><th>現在在哪</th><th></th></tr>${d.batches.map(b=>`<tr>
    <td>${esc(b.id)}</td><td>${esc(b.created.replace("T"," "))}</td><td>${esc(b.sources.join("；"))}</td><td>${b.mode==="copy"?"複製":"搬移"} ${b.videos}</td>
    <td>${b.undone?'<span class="hint">已全部還原</span>':Object.entries(b.now).map(([k,v])=>`${NOW[k]||k} ${v}`).join("、")||'<span class="hint">不在工作區</span>'}</td>
    <td>${b.undone?"":`<button data-undo="${esc(b.id)}">整批還原…</button>`}</td></tr>`).join("")}</table></div>`;
  box.querySelectorAll("button[data-undo]").forEach(btn=>btn.onclick=async()=>{
    const id=btn.dataset.undo; const [ok,res]=await post("/api/undo_batch",{batch:id,apply:false});
    if(!ok){alert(res.error||"失敗");return;}
    const lines=res.plans.map(p=>`${p.kind}（${p.run}）：${p.n} 筆${p.partial?"（只還原屬於這批的）":""}${p.skipped.length?`，略過 ${p.skipped.length} 筆`:""}`);
    const skipped=res.plans.flatMap(p=>p.skipped).slice(0,5).map(([s,why])=>`  略過 ${s}：${why}`);
    if(!confirm(`整批還原 ${id}？會依相反順序退回：\n`+lines.join("\n")+(skipped.length?"\n"+skipped.join("\n"):"")+"\n\n搬移的會搬回原位置，複製的會刪掉複本。"))return;
    const [ok2,res2]=await post("/api/undo_batch",{batch:id,apply:true}); if(!ok2)alert(res2.error||"失敗"); poll();
  });
}
let pollTimer=null;
async function poll(){
  if(pollTimer){clearTimeout(pollTimer);pollTimer=null;}
  const d=await (await fetch("/api/status")).json();
  $("#stages").textContent=d.stages.map(([l,n])=>`${l} ${n}`).join("　");
  $("#paths").textContent=d.paths.root?`${d.paths.root}（作者資料夾在這裡；分類中的在 ${d.paths.workspace}）`:`收藏庫 ${d.paths.library}；工作區 ${d.paths.workspace}`;
  if(!$("input[name=mode]:checked")) $(`input[name=mode][value=${d.default_mode}]`).checked=true;
  const j=d.job; const st=$("#state"); st.className="state "+j.state;
  st.textContent={idle:"",running:"執行中",waiting:"等你確認",done:"完成",cancelled:"已取消",error:"失敗"}[j.state]||j.state; $("#step").textContent=(j.state==="running"||j.state==="waiting")?j.step:"";
  $("#start").disabled=j.state==="running"||j.state==="waiting";
  $("#rematch").disabled=$("#start").disabled||!d.unknown_authors; $("#unk").textContent=d.unknown_authors?`收藏庫 作者未知 目前 ${d.unknown_authors} 部；每批分類完也會自動比對一次`:"收藏庫 作者未知 目前沒有影片";
  $("#start").textContent=(!picked.size&&d.pending_in_workspace&&!(j.state==="running"||j.state==="waiting"))?`接著處理工作區的 ${d.pending_in_workspace} 部`:"開始分類";
  renderPlan(j);
  const log=$("#log"); const atBottom=log.scrollTop+log.clientHeight>=log.scrollHeight-4; log.textContent=j.log.join("\n"); if(atBottom)log.scrollTop=log.scrollHeight;
  const r=$("#result");
  if(j.state==="done"&&j.counts&&"restored" in j.counts) r.innerHTML=`已整批還原 <b>${j.counts.restored}</b> 筆`;
  else if(j.state==="done"&&j.counts&&!("library" in j.counts)) r.innerHTML=j.counts.rematched?`<a href="/review">${j.counts.rematched} 部未知作者找到相似作者，等你確認 →</a>`:"沒有夠像的作者";
  else if(j.state==="done"&&j.counts) r.innerHTML=`已確定 → 04 收藏庫 <b>${j.counts.library}</b> 部；可能誤判 → <a href="/review"><b>${j.counts.review}</b> 部等人工確認 →</a>${j.counts.rematched?`<div class="hint">另外 ${j.counts.rematched} 部原本未知作者的影片找到相似作者，也在人工確認頁</div>`:""}${j.batch?`<div class="hint">批次 ${esc(j.batch)}，可在下方歷史批次整批還原</div>`:""}`;
  else if(j.state==="error") r.innerHTML=`<span class="state error">${esc(j.error)}</span>`;
  else r.innerHTML=d.incomplete.length?`<span class="state error">有中斷的紀錄 ${esc(d.incomplete.join(", "))}，請先在命令列執行 vsort resume</span>`:"";
  if((j.state==="done"||j.state==="cancelled"||j.state==="error")&&(lastState==="running"||lastState==="waiting")){browse(cur);loadBatches();}
  lastState=j.state;
  pollTimer=setTimeout(poll, j.state==="running"?1500:j.state==="waiting"?3000:5000);
}
(async()=>{const d=await (await fetch("/api/status")).json(); let start=""; try{start=localStorage.getItem("vsort.dir")||"";}catch(e){} if(!start&&d.start_dirs.length)start=d.start_dirs[0]; await browse(start); poll(); loadBatches();})();
</script></body></html>
"""
