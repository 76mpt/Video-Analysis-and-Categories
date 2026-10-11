"""動畫類型（MMD／3D／2D，可複選）：比較 CLIP、SigLIP 零樣本與自訓小分類器，並訓練最好的那個。

資料：dict\\style_labels.csv（標註網頁產生）。每部影片用快取的 16 張關鍵幀。
特徵：CLIP ViT-L/14、SigLIP2 so400m 的圖片向量（每部影片取平均），以及 PixAI 的畫風標籤機率。
小分類器：每個類型各一個邏輯迴歸（可複選），5 折交叉驗證評估。
訓練好的模型存成 models\\style_classifier.json，設定檔 ai.style_model 設成 trained 就會使用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vsort.config import Config
from vsort.frames import extract_cached
from vsort.naming import ANIM_ORDER
from vsort.tagger import MODELS_DIR

CLIP_REPO = "openai/clip-vit-large-patch14"
SIGLIP_REPO = "google/siglip2-so400m-patch14-384"
PIXAI_REPO = "pixai-labs/pixai-tagger-v1.0"
CLASSIFIER_PATH = MODELS_DIR / "style_classifier.json"

CLIP_PROMPTS = {
    "MMD": "a MikuMikuDance MMD 3D toon-shaded anime character render",
    "Koikatsu": "a screenshot of the Koikatsu anime 3D game by Illusion",
    "VAM": "a realistic 3D render from Virt-A-Mate (VaM)",
    "3D": "a 3D CG render of a character made in Blender or Source Filmmaker",
    "2D": "a 2D hand-drawn anime illustration",
    "Audio": "a single still cover image with text, like an audio drama cover",
}
SIGLIP_PROMPTS = {
    "MMD": "a screenshot of a MikuMikuDance (MMD) video with a toon-shaded 3D anime model",
    "Koikatsu": "a screenshot of a Koikatsu (Illusion) 3D anime game video",
    "VAM": "a screenshot of a Virt-A-Mate (VaM) video with a realistic 3D character",
    "3D": "a screenshot of a 3D CGI animation rendered in Blender or Source Filmmaker",
    "2D": "a screenshot of a traditional 2D hand-drawn anime",
    "Audio": "a static cover image of an audio drama with no animation",
}


# ---------- 特徵擷取 ----------

class Embedder:
    """把一部影片的關鍵幀轉成一個向量（取平均）。結果快取在 _cache/emb。"""

    kind = ""

    def __init__(self, cache: Path):
        self.cache = cache / "emb" / self.kind
        self.cache.mkdir(parents=True, exist_ok=True)

    def video(self, key: str, images: list) -> np.ndarray:
        p = self.cache / f"{key}.npy"
        if p.exists():
            return np.load(p)
        v = self.embed(images).mean(axis=0)
        np.save(p, v)
        return v

    def embed(self, images: list) -> np.ndarray:
        raise NotImplementedError


class ClipEmbedder(Embedder):
    kind = "clip"

    def __init__(self, cache: Path, repo: str = CLIP_REPO):
        super().__init__(cache)
        self.repo = repo
        self._m = None

    def _load(self):
        if self._m is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self._m = CLIPModel.from_pretrained(self.repo, dtype=self.dtype).to(self.dev).eval()
            self.proc = CLIPProcessor.from_pretrained(self.repo)

    def embed(self, images):
        self._load()
        inp = self.proc(images=images, return_tensors="pt").to(self.dev)
        with self.torch.inference_mode():
            f = self._m.get_image_features(pixel_values=inp["pixel_values"].to(self.dtype))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def zero_shot(self, images, prompts=CLIP_PROMPTS) -> dict[str, float]:
        self._load()
        torch = self.torch
        txt = self.proc(text=list(prompts.values()), return_tensors="pt", padding=True).to(self.dev)
        inp = self.proc(images=images, return_tensors="pt").to(self.dev)
        with torch.inference_mode():
            logits = self._m(input_ids=txt["input_ids"], attention_mask=txt["attention_mask"],
                             pixel_values=inp["pixel_values"].to(self.dtype)).logits_per_image
            p = logits.softmax(-1).float().mean(0).cpu().numpy()
        return dict(zip(prompts, map(float, p)))


class SiglipEmbedder(Embedder):
    kind = "siglip"

    def __init__(self, cache: Path, repo: str = SIGLIP_REPO):
        super().__init__(cache)
        self.repo = repo
        self._m = None

    def _load(self):
        if self._m is None:
            import torch
            from transformers import AutoModel, AutoProcessor
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self._m = AutoModel.from_pretrained(self.repo, dtype=self.dtype).to(self.dev).eval()
            self.proc = AutoProcessor.from_pretrained(self.repo)

    def embed(self, images):
        self._load()
        inp = self.proc(images=images, return_tensors="pt").to(self.dev)
        with self.torch.inference_mode():
            f = self._m.get_image_features(pixel_values=inp["pixel_values"].to(self.dtype))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def zero_shot(self, images, prompts=SIGLIP_PROMPTS) -> dict[str, float]:
        """SigLIP 每個描述各自給一個機率（sigmoid），本來就能複選。"""
        self._load()
        torch = self.torch
        txt = self.proc(text=list(prompts.values()), return_tensors="pt", padding="max_length",
                        max_length=64).to(self.dev)
        inp = self.proc(images=images, return_tensors="pt").to(self.dev)
        with torch.inference_mode():
            out = self._m(input_ids=txt["input_ids"], pixel_values=inp["pixel_values"].to(self.dtype))
            p = torch.sigmoid(out.logits_per_image).float().mean(0).cpu().numpy()
        return dict(zip(prompts, map(float, p)))


class PixaiStyleEmbedder(Embedder):
    """PixAI 的「畫風」類別標籤機率（含 3d、mikumikudance_(medium)、koikatsu_(medium)、blender_(medium) 等）。"""
    kind = "pixai_style"

    def __init__(self, cache: Path, repo: str = PIXAI_REPO):
        super().__init__(cache)
        self.repo = repo
        self._t = None

    def embed(self, images):
        import torch
        if self._t is None:
            from vsort.tagger import PixAITagger
            self._t = PixAITagger(self.repo)
            cfg = self._t.model.config
            st = 0
            for cat, count in cfg.tags_split:
                if cat == "style":
                    self.idx = list(range(st, st + count))
                st += count
        x = self._t.proc(images)["pixel_values"].to(self._t.device)
        x = x.half() if self._t.device == "cuda" else x
        rows = []
        for i in range(0, len(x), 2):
            with torch.inference_mode():
                rows.append(self._t.model(x[i:i + 2]).sigmoid().float().cpu().numpy()[:, self.idx])
        return np.concatenate(rows)


EMBEDDERS = {"clip": ClipEmbedder, "siglip": SiglipEmbedder, "pixai_style": PixaiStyleEmbedder}


# ---------- 評估與訓練 ----------

def _dataset(cfg: Config, frames: int = 16):
    from vsort.labeling import load_labels
    labels = {k: v for k, v in load_labels(cfg).items() if not v.get("skip") and v["labels"]}
    cache = cfg.layout.workspace / "_cache"
    items = []
    for key, v in labels.items():
        p = Path(v["path"])
        fdir = cache / "frames" / key
        if not (fdir / "frames.json").exists() and not p.exists():
            continue
        items.append((key, p, v["labels"], fdir))
    y = np.array([[1 if a in labs else 0 for a in ANIM_ORDER] for _, _, labs, _ in items])
    return items, y, cache


def _metrics(y: np.ndarray, pred: np.ndarray, top1: np.ndarray | None = None) -> dict:
    out = {"exact": round(float((pred == y).all(axis=1).mean()), 3)}
    # 寬鬆正確：你勾的都有標到，多標的只有 MMD（你說 MMD 多標可以接受）
    mmd = ANIM_ORDER.index("MMD")
    extra = (pred == 1) & (y == 0)
    extra[:, mmd] = False
    out["lenient"] = round(float((((pred == 1) | (y == 0)).all(axis=1) & ~extra.any(axis=1)).mean()), 3)
    if top1 is not None:
        out["top1"] = round(float(y[np.arange(len(y)), top1].mean()), 3)
    for j, a in enumerate(ANIM_ORDER):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((pred[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        out[f"F1_{a}"] = round(f1, 3) if f1 is not None else None
    return out


def _cv_logreg(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> tuple[np.ndarray, float]:
    """回傳交叉驗證的預測機率，以及選到的正則化強度 C。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    best = None
    for C in (0.01, 0.1, 1.0):
        prob = np.zeros(y.shape, dtype=float)
        kf = KFold(n_splits=min(folds, len(X)), shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            for j in range(y.shape[1]):
                if y[tr, j].min() == y[tr, j].max():  # 這一折訓練資料只有一種答案
                    prob[te, j] = y[tr, j].mean()
                    continue
                m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr, j])
                prob[te, j] = m.predict_proba(sc.transform(X[te]))[:, 1]
        score = ((prob >= 0.5).astype(int) == y).all(axis=1).mean()
        if best is None or score > best[0]:
            best = (score, prob, C)
    return best[1], best[2]


def evaluate(cfg: Config, frames: int = 16, features=("clip", "siglip", "pixai_style"),
             progress=print) -> dict:
    items, y, cache = _dataset(cfg, frames)
    if len(items) < 10:
        raise RuntimeError(f"標註樣本只有 {len(items)} 部，至少需要 10 部（建議 100 部以上）。")
    counts = {a: int(y[:, j].sum()) for j, a in enumerate(ANIM_ORDER)}
    progress(f"樣本 {len(items)} 部：" + "、".join(f"{a} {n}" for a, n in counts.items()))
    images = {}
    for key, p, _, fdir in items:
        images[key] = [im for _, im in extract_cached(p, frames, fdir)]
    report: dict = {"n": len(items), "counts": counts, "methods": {}}
    feats: dict[str, np.ndarray] = {}
    for name in features:
        emb = EMBEDDERS[name](cache)
        progress(f"計算 {name} 特徵…")
        feats[name] = np.stack([emb.video(k, images[k]) for k, *_ in items])
        if name in ("clip", "siglip"):
            zs = np.array([[emb.zero_shot(images[k])[a] for a in ANIM_ORDER] for k, *_ in items])
            top1 = zs.argmax(axis=1)
            if name == "clip":
                pred = np.eye(len(ANIM_ORDER), dtype=int)[top1]
            else:  # SigLIP：最高的那個，加上機率接近最高的（可複選）
                pred = ((zs >= zs.max(axis=1, keepdims=True) * 0.5)).astype(int)
            report["methods"][f"{name} 零樣本"] = _metrics(y, pred, top1)
        del emb
        _free_gpu()
    for name, X in feats.items():
        prob, C = _cv_logreg(X, y)
        report["methods"][f"小分類器（{name} 特徵）"] = {**_metrics(y, (prob >= 0.5).astype(int), prob.argmax(axis=1)), "C": C}
    if len(feats) > 1:
        X = np.concatenate(list(feats.values()), axis=1)
        prob, C = _cv_logreg(X, y)
        report["methods"]["小分類器（全部特徵）"] = {**_metrics(y, (prob >= 0.5).astype(int), prob.argmax(axis=1)), "C": C}
    return report


def train(cfg: Config, feature: str, C: float, frames: int = 16) -> Path:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    items, y, cache = _dataset(cfg, frames)
    names = feature.split("+")
    embs = [EMBEDDERS[n](cache) for n in names]
    X = np.stack([np.concatenate([e.video(k, [im for _, im in extract_cached(p, frames, fdir)]) for e in embs])
                  for k, p, _, fdir in items])
    sc = StandardScaler().fit(X)
    heads = {}
    for j, a in enumerate(ANIM_ORDER):
        if y[:, j].min() == y[:, j].max():
            heads[a] = {"constant": float(y[:, j].mean())}
            continue
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y[:, j])
        heads[a] = {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_PATH.write_text(json.dumps({"features": names, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                                           "heads": heads, "n": len(items)}), encoding="utf-8")
    return CLASSIFIER_PATH


class TrainedStyleClassifier:
    """執行時用：算特徵 → 套訓練好的邏輯迴歸 → 每個類型各自的機率（可複選）。"""

    kind = "classifier"

    def __init__(self, cache: Path, path: Path = CLASSIFIER_PATH):
        d = json.loads(path.read_text(encoding="utf-8"))
        self.embs = [EMBEDDERS[n](cache) for n in d["features"]]
        self.mean, self.scale = np.array(d["mean"]), np.array(d["scale"])
        self.heads = d["heads"]

    def predict_video(self, key: str, images: list) -> dict[str, float]:
        x = (np.concatenate([e.video(key, images) for e in self.embs]) - self.mean) / self.scale
        out = {}
        for a, h in self.heads.items():
            out[a] = h["constant"] if "constant" in h else float(1 / (1 + np.exp(-(x @ np.array(h["coef"]) + h["intercept"]))))
        return out


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
