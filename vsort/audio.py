"""聲音特徵（類別判定的輔助訊號）：Dance 多半是有穩定節拍的音樂，Scene 多半是人聲為主、音樂少；
Mix 則是兩種段落交替。把音軌切成 16 段，每段算幾個簡單的聲學特徵，再取整部的平均、起伏（標準差）、最大、最小，
交給類別分類器當額外特徵。不用下載任何模型，只用 ffmpeg 解碼與 librosa。結果快取在 Workspace/_cache/audio。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
WINDOWS = 16
CLIP_SECONDS = 6  # 每段只分析中間 6 秒，整部約 100 秒的聲音，每部不到 1 秒
MAX_SECONDS = 20 * 60  # 太長的影片只看前 20 分鐘
FEATURE_VERSION = 1
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PER_WINDOW = ["rms", "flatness", "centroid", "percussive", "onset", "beat_clarity", "voice_band", "zcr"]
SUMMARIES = ["mean", "std", "min", "max"]
N_MFCC = 13
FEATURE_NAMES = [f"{k}_{s}" for k in PER_WINDOW for s in SUMMARIES] + ["silent_share", "has_audio"] + \
    [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]


def decode(path: Path, sr: int = SR, max_seconds: int = MAX_SECONDS) -> np.ndarray | None:
    """用 ffmpeg 把音軌解成單聲道 16 kHz 的浮點陣列；沒有音軌回傳 None。"""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=_NO_WINDOW)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or len(out.stdout) < sr:  # 不到 1 秒也當作沒有
        return None
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _window_features(y: np.ndarray, sr: int) -> list[float]:
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    rms = float(librosa.feature.rms(S=S, frame_length=1024).mean())
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    cent = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()) / (sr / 2)
    H, P = librosa.decompose.hpss(S, kernel_size=15)  # 預設 31 太慢（每段 0.5 秒），15 幾乎一樣準
    perc = float(P.sum() / (H.sum() + P.sum() + 1e-9))
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2), sr=sr, hop_length=256)
    onset_mean = float(onset.mean())
    # 節拍清晰度：起音包絡的自相關在 0.3～1 秒（60～200 BPM）範圍內的最大峰值
    ac = librosa.autocorrelate(onset - onset.mean()) if len(onset) > 8 else np.zeros(2)
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(0.3 * sr / 256), int(1.0 * sr / 256)
    clarity = float(ac[lo:hi].max()) if len(ac) > hi else 0.0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = (freqs >= 300) & (freqs <= 3400)
    voice = float(S[band].sum() / (S.sum() + 1e-9))
    zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())
    return [rms, flat, cent, perc, onset_mean, clarity, voice, zcr]


def features(y: np.ndarray | None, sr: int = SR, windows: int = WINDOWS) -> list[float]:
    """整部影片的聲音特徵向量（長度固定＝len(FEATURE_NAMES)）。沒有音軌時全部為 0、has_audio=0。"""
    import librosa
    n = len(FEATURE_NAMES)
    if y is None or len(y) < sr:
        return [0.0] * n
    seg = len(y) // windows
    clip = min(seg, CLIP_SECONDS * sr)
    clips = [y[i * seg + (seg - clip) // 2: i * seg + (seg - clip) // 2 + clip] for i in range(windows)]
    per = np.array([_window_features(c, sr) for c in clips])
    out: list[float] = []
    for j in range(per.shape[1]):
        col = per[:, j]
        out += [float(col.mean()), float(col.std()), float(col.min()), float(col.max())]
    silent = float((per[:, 0] < 0.005).mean())
    out += [silent, 1.0]
    mf = librosa.feature.mfcc(y=np.concatenate(clips), sr=sr, n_mfcc=N_MFCC)  # 音色：用同樣這 16 段
    out += mf.mean(axis=1).tolist() + mf.std(axis=1).tolist()
    return out


def cached_features(video: Path, cache_dir: Path, key: str) -> list[float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}_v{FEATURE_VERSION}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if len(d.get("x", [])) == len(FEATURE_NAMES):
                return d["x"]
        except (OSError, ValueError):
            pass
    x = features(decode(video))
    p.write_text(json.dumps({"version": FEATURE_VERSION, "x": x}), encoding="utf-8")
    return x
