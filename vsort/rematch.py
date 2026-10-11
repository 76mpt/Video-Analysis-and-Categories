"""未知作者的影片之後再認：用畫風指紋（SigLIP 向量）和「現在已知」的作者比對。

收藏庫 `作者未知` 裡的影片，每次有新影片歸檔（或你人工確認）後，已知作者的樣本就變多；
重新比對時，夠像的（相似度 ≥ min_similarity 且領先第二名 ≥ min_margin）就搬到 05 人工確認，
作者欄預填成最像的那位、附相似度，你看一眼按確認就歸到那位作者底下。

門檻怎麼來：你標過的 421 部（70 位作者）留一法：相似度 ≥ 0.5 且領先 ≥ 0.1 時，猜對 98%、涵蓋 69%；
因為最後還是你確認，門檻放寬一點沒關係。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vsort import sidecar as sc
from vsort.analyze import UNKNOWN_AUTHOR_DIR, file_key
from vsort.config import Config
from vsort.journal import Op, Plan
from vsort.naming import build_final_name, parse_final_name, unique_path
from vsort.process import _videos

MIN_SIMILARITY = 0.5
MIN_MARGIN = 0.1


def _ensure_embedding(cfg: Config, video: Path, key: str, embedder=None):
    """沒有畫風向量（例如還沒分析過）就從快取的關鍵幀算一個。回傳 embedder（可能剛載入）。"""
    emb_path = cfg.layout.workspace / "_cache" / "emb" / "siglip" / f"{key}.npy"
    if emb_path.exists():
        return embedder
    from vsort.frames import extract_cached
    from vsort.style_train import SiglipEmbedder
    embedder = embedder or SiglipEmbedder(cfg.layout.workspace / "_cache")
    imgs = [im for _, im in extract_cached(video, 16, cfg.layout.workspace / "_cache" / "frames" / key)]
    if imgs:
        embedder.video(key, imgs)
    return embedder


def plan_rematch(cfg: Config, min_similarity: float = MIN_SIMILARITY, min_margin: float = MIN_MARGIN,
                 index=None, progress=None) -> tuple[Plan, int]:
    """回傳（計畫, 檢查了幾部）。計畫裡每一筆都是 作者未知 → 05_Review，作者預填。"""
    from vsort.style_author import StyleAuthorIndex
    lay = cfg.layout
    naming = cfg.naming
    plan = Plan(kind="rematch")
    targets = _videos(lay.library / UNKNOWN_AUTHOR_DIR, cfg)
    if not targets:
        return plan, 0
    index = index or StyleAuthorIndex(cfg)
    taken: set[str] = set()
    embedder = None
    for video in targets:
        key = file_key(video)
        try:
            embedder = _ensure_embedding(cfg, video, key, embedder)
        except Exception as exc:  # 算不出向量就跳過這部
            plan.skipped.append((str(video), f"算不出畫風向量：{exc}"))
            continue
        sugg = index.suggest(key, k=3)
        if progress:
            progress(video.name, sugg)
        if not sugg:
            continue
        best, s1 = sugg[0]
        s2 = sugg[1][1] if len(sugg) > 1 else 0.0
        if s1 < min_similarity or s1 - s2 < min_margin:
            continue
        prev = sc.read(video) or {}
        info = dict(prev.get("info") or parse_final_name(video.name) or {"title": video.stem})
        info["author"] = best
        data = dict(prev)
        reason = f"畫風指紋與作者 {best} 相似（{s1:.2f}；第二名 {sugg[1][0]} {s2:.2f}），請確認" if len(sugg) > 1 \
            else f"畫風指紋與作者 {best} 相似（{s1:.2f}），請確認"
        data.update(info=info, certainty="可能誤判", reasons=[reason], stage="review", style_authors=[list(x) for x in sugg])
        cands = dict(data.get("candidates") or {})
        cands["authors"] = [list(x) for x in sugg]
        data["candidates"] = cands
        data.setdefault("history", []).append(
            {"t": datetime.now().isoformat(timespec="seconds"), "step": "rematch", "stage": "review"})
        name = build_final_name(info, video.suffix, naming["illegal_char_style"], naming["max_name_length"])
        dst = unique_path(lay.review / name, taken)
        plan.ops.append(Op(action="move", src=str(video), dst=str(dst), certainty="可能誤判", note=reason,
                           sidecar_mode="write", sidecar=data, sidecar_prev=prev or None))
    return plan, len(targets)
