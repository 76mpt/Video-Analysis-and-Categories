"""關鍵幀擷取（F18）：用 ffmpeg 在影片中平均取幾張畫面。"""

from __future__ import annotations

import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, timeout=60, creationflags=_NO_WINDOW)
    try:
        return float(json.loads(out.stdout or b"{}")["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0


def grab(path: Path, t: float, max_side: int = 768) -> Image.Image | None:
    vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    # -skip_frame nokey：只解碼關鍵畫格，4K 影片快將近一倍；取到的畫面離指定時間差幾秒，對分類沒有影響
    cmd = ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
           "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=_NO_WINDOW)
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        return Image.open(io.BytesIO(out.stdout)).convert("RGB")
    except OSError:
        return None


def extract(path: Path, count: int = 16, max_side: int = 768) -> list[tuple[float, Image.Image]]:
    """平均取 count 張，避開片頭片尾各 3%。"""
    dur = duration(path)
    if dur <= 0:
        img = grab(path, 0, max_side)
        return [(0.0, img)] if img else []
    start, end = dur * 0.03, dur * 0.97
    step = (end - start) / max(count, 1)
    times = [start + step * (i + 0.5) for i in range(count)]
    with ThreadPoolExecutor(max_workers=4) as pool:  # 同時開 4 個 ffmpeg
        imgs = list(pool.map(lambda t: grab(path, t, max_side), times))
    return [(t, img) for t, img in zip(times, imgs) if img is not None]


def contact_sheet(frames: list[tuple[float, Image.Image]], cols: int = 4, cell: int = 320) -> Image.Image:
    """把關鍵幀拼成一張預覽圖，給人工確認時看。"""
    rows = max(1, (len(frames) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    for i, (_, img) in enumerate(frames):
        im = img.copy()
        im.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    return sheet


def extract_cached(path: Path, count: int, cache_dir: Path) -> list[tuple[float, Image.Image]]:
    """同 extract，但把關鍵幀存成 JPEG 快取；換模型重跑、標註、訓練時不用再讀影片。"""
    marker = cache_dir / "frames.json"
    if marker.exists():
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("count") == count:
                return [(t, Image.open(cache_dir / f"{i:02d}.jpg").convert("RGB"))
                        for i, t in enumerate(meta["times"])]
        except (OSError, ValueError, KeyError, AttributeError):
            pass
    frames = extract(path, count)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, img) in enumerate(frames):
        img.save(cache_dir / f"{i:02d}.jpg", quality=92)
    marker.write_text(json.dumps({"count": count, "times": [t for t, _ in frames]}), encoding="utf-8")
    return frames
