"""動漫圖片標籤模型（F19、F21 的基礎）：WD Tagger v3（SmilingWolf，開源，本機 GPU 執行）。

一個模型同時給出：
- 角色標籤（danbooru 角色名，例如 raiden_shogun）
- 一般標籤（例如 dancing、3d）
- 分級（general / sensitive / questionable / explicit）
模型下載到專案資料夾裡的 models。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from vsort.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

DEFAULT_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"


@dataclass
class Labels:
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]


def _load_labels(path: Path) -> Labels:
    names, rating, general, character = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            names.append(row["name"])
            cat = int(row["category"])
            (rating if cat == 9 else character if cat == 4 else general).append(i)
    return Labels(names, rating, general, character)


class Tagger:
    def __init__(self, repo: str = DEFAULT_REPO, device: str | None = None, batch_size: int = 8):
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.labels = _load_labels(Path(hf_hub_download(repo, "selected_tags.csv")))
        model = timm.create_model("hf-hub:" + repo, pretrained=True)
        model.eval()
        if self.device == "cuda":
            model = model.half()
        self.model = model.to(self.device)
        cfg = model.pretrained_cfg
        self.size = cfg.get("input_size", (3, 448, 448))[-1]
        self.mean = cfg.get("mean", (0.5, 0.5, 0.5))
        self.std = cfg.get("std", (0.5, 0.5, 0.5))

    def _prep(self, img):
        from PIL import Image
        import numpy as np

        # 補白成正方形再縮放（WD Tagger 的訓練方式）
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        canvas = canvas.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        arr = arr.transpose(2, 0, 1)[::-1].copy()  # RGB→BGR，模型是用 BGR 訓練的
        return arr

    def predict(self, images: list) -> list[dict[str, float]]:
        """回傳每張圖的 {標籤: 機率}（只保留機率 ≥ 0.05 的，減少記憶體）。"""
        import numpy as np

        torch = self.torch
        out: list[dict[str, float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([self._prep(im) for im in images[i:i + self.batch_size]])
            x = torch.from_numpy(batch).to(self.device)
            if self.device == "cuda":
                x = x.half()
            with torch.inference_mode():
                probs = torch.sigmoid(self.model(x)).float().cpu().numpy()
            for row in probs:
                idx = np.nonzero(row >= 0.05)[0]
                out.append({self.labels.names[j]: float(row[j]) for j in idx} |
                           {self.labels.names[j]: float(row[j]) for j in self.labels.rating})
        return out

    def character_set(self) -> set[str]:
        return {self.labels.names[i] for i in self.labels.character}


STYLE_PROMPTS = {
    "MMD": "a MikuMikuDance MMD 3D toon-shaded anime character render",
    "3D": "a realistic 3D CG render of a character, made in Blender or Koikatsu",
    "2D": "a 2D hand-drawn anime illustration",
}
DEFAULT_STYLE_MODEL = "openai/clip-vit-large-patch14"


class StyleClassifier:
    """動畫類型（F28）：用 CLIP 零樣本分類判斷畫面像 MMD、3D 還是 2D。

    實測（你的影片）：MMD 與 Koikatsu／Blender 大致分得開，但部分 MMD 會偏向 3D，
    所以信心不夠時會送人工確認；檔名或網站標籤寫明時以那個為準。
    """

    def __init__(self, repo: str = DEFAULT_STYLE_MODEL, prompts: dict | None = None, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(repo, dtype=dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(repo)
        self.labels = list((prompts or STYLE_PROMPTS).keys())
        self.text = self.proc(text=list((prompts or STYLE_PROMPTS).values()), return_tensors="pt",
                              padding=True).to(self.device)
        self.dtype = dtype

    def predict(self, images: list) -> list[dict[str, float]]:
        torch = self.torch
        out = []
        for i in range(0, len(images), 16):
            inp = self.proc(images=images[i:i + 16], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=self.text["input_ids"], attention_mask=self.text["attention_mask"],
                                    pixel_values=inp["pixel_values"].to(self.dtype)).logits_per_image
                probs = logits.softmax(-1).float().cpu().numpy()
            out.extend({lab: float(p) for lab, p in zip(self.labels, row)} for row in probs)
        return out


RATING_ALIASES = {"rating:g": "general", "rating:s": "sensitive", "rating:q": "questionable", "rating:e": "explicit",
                  "rating_general": "general", "rating_sensitive": "sensitive",
                  "rating_questionable": "questionable", "rating_explicit": "explicit"}


class PixAITagger:
    """PixAI Tagger v1.0（pixai-labs）：較新、角色庫較大；輸入 1008px，比 WD 慢。

    模型附的程式碼（tagger_pipeline.py）已檢查過，只有模型定義，沒有連網或執行外部指令。
    """

    def __init__(self, repo: str = "pixai-labs/pixai-tagger-v1.0", device: str | None = None, batch_size: int = 2):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.proc = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
        cfg = self.model.config
        self.names = [RATING_ALIASES.get(t, t) for t in cfg.tags]
        self.chars: set[str] = set()
        st = 0
        for cat, count in cfg.tags_split:
            if cat == "character":
                self.chars = set(self.names[st:st + count])
            st += count
        self.rating_idx = [i for i, t in enumerate(cfg.tags) if t in RATING_ALIASES]

    def predict(self, images: list) -> list[dict[str, float]]:
        import numpy as np
        torch = self.torch
        out = []
        for i in range(0, len(images), self.batch_size):
            x = self.proc(images[i:i + self.batch_size])["pixel_values"].to(self.device)
            x = x.half() if self.device == "cuda" else x
            with torch.inference_mode():
                probs = self.model(x).sigmoid().float().cpu().numpy()
            for row in probs:
                idx = np.nonzero(row >= 0.05)[0]
                out.append({self.names[j]: float(row[j]) for j in idx} |
                           {self.names[j]: float(row[j]) for j in self.rating_idx})
        return out

    def character_set(self) -> set[str]:
        return self.chars


class CamieTagger:
    """Camie Tagger v2（Camais03）：7 萬個標籤，ONNX 版本，輸入 512px。"""

    def __init__(self, repo: str = "Camais03/camie-tagger-v2", batch_size: int = 8):
        import json as _json

        import torch  # noqa: F401  先載入 PyTorch，onnxruntime 才能用它附帶的 CUDA 12 DLL
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        ort.set_default_logger_severity(3)

        meta = _json.loads(Path(hf_hub_download(repo, "camie-tagger-v2-metadata.json")).read_text(encoding="utf-8"))
        tm = meta["dataset_info"]["tag_mapping"]
        idx_to_tag = tm["idx_to_tag"]
        self.names = [RATING_ALIASES.get(idx_to_tag[str(i)], idx_to_tag[str(i)]) for i in range(len(idx_to_tag))]
        cats = tm["tag_to_category"]
        self.chars = {RATING_ALIASES.get(t, t) for t, c in cats.items() if c == "character"}
        self.rating_idx = [i for i, t in enumerate(self.names) if t in RATING_ALIASES.values()]
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(hf_hub_download(repo, "camie-tagger-v2.onnx"), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.batch_size = batch_size

    def _prep(self, img):
        import numpy as np
        from PIL import Image

        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        arr = np.asarray(canvas.resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return arr.transpose(2, 0, 1)

    def predict(self, images: list) -> list[dict[str, float]]:
        import numpy as np
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([self._prep(im) for im in images[i:i + self.batch_size]]).astype(np.float32)
            res = dict(zip(self.outputs, self.sess.run(None, {self.input_name: batch})))
            logits = res.get("refined_predictions", next(iter(res.values())))
            probs = 1 / (1 + np.exp(-logits))
            for row in probs:
                idx = np.nonzero(row >= 0.05)[0]
                out.append({self.names[j]: float(row[j]) for j in idx} |
                           {self.names[j]: float(row[j]) for j in self.rating_idx})
        return out

    def character_set(self) -> set[str]:
        return self.chars


def make_tagger(name: str):
    """依設定檔的 ai.model 建立標籤模型：WD 系列（repo 名稱）、pixai、camie。"""
    n = name.lower()
    if "pixai" in n:
        return PixAITagger(name if "/" in name else "pixai-labs/pixai-tagger-v1.0")
    if "camie" in n:
        return CamieTagger(name if "/" in name else "Camais03/camie-tagger-v2")
    return Tagger(name)
