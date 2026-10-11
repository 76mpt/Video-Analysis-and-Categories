"""類別（Dance／Mix／Scene）分類器：把畫面標籤的比例和 Qwen3-VL 的判斷合在一起，用你標的類別訓練。

實測（你標的 151 部、5 折交叉驗證）：
- Qwen3-VL 整部描述 73%、逐張判斷 80%、影片模式 72%、畫面標籤 80%
- 合併「畫面標籤＋Qwen 逐張＋Qwen 影片模式」86%；信心 ≥ 0.8 的約 66%，其中正確 91%
- 加上更細的逐張是非題提示詞（v2）反而降到 74%，所以沒有採用
Qwen 每部約 3.5 秒（逐張 2.5 秒＋影片模式 1 秒），比原本整部描述的 5.4 秒快。
模型存成 models\\category_classifier.json，設定檔 ai.category_model 設成 combined 就會使用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vsort.tagger import MODELS_DIR

CATEGORY_CLASSIFIER = MODELS_DIR / "category_classifier.json"
# 第二段：信心不足時才加跑備援版逐張判斷，再用這個分類器判斷（251 部實測：自動歸檔 47% → 50%，Mix 53% → 57%）
FALLBACK_CLASSIFIER = MODELS_DIR / "category_classifier_fallback.json"
FALLBACK_FEATURE_SET = "tags+frames_v1+video_v1+frames_v3"
CATS = ["Dance", "Mix", "Scene"]
FEATURE_SET = "tags+frames_v1+video_v1"  # 預設；加 +frames_v3（備援版）在 251 部只多 1%，每部卻多約 10 秒
MIX_WEIGHT = 1.8  # 訓練時加重 Mix：251 部實測 Mix 找回率 33% → 53%，總正確率不變
AUDIO = "+audio_v1"  # 聲音特徵（vsort/audio.py）：加在任一特徵組合後面
FEATURE_SETS = ("tags+frames_v1+video_v1", "tags+frames_v1+video_v1+frames_v3",
                "tags+frames_v1+video_v1+audio_v1", "tags+frames_v1+video_v1+frames_v3+audio_v1")
FEATURES = ["場景畫面比例", "跳舞畫面比例", "露骨畫面比例",
            "Qwen 逐張：跳舞", "Qwen 逐張：場景", "Qwen 逐張：其他", "Qwen 逐張：跳舞與場景切換",
            "Qwen 影片模式=Dance", "Qwen 影片模式=Mix", "Qwen 影片模式=Scene", "Qwen 影片模式：場景比例"]


def features_v2(scene_ratio: float, dance_ratio: float, explicit_ratio: float, frames_stats: dict | None,
                video: dict | None) -> list[float]:
    """目前使用的特徵：畫面標籤比例＋Qwen 逐張判斷＋Qwen 影片模式。"""
    fs = frames_stats or {}
    vd = video or {}
    vf = vd.get("scene_fraction")
    try:
        vf = min(max(float(vf), 0.0), 1.0)
    except (TypeError, ValueError):
        vf = 0.5
    return [scene_ratio, dance_ratio, explicit_ratio, *(float(fs.get(k, 0.0)) for k in ("D", "S", "O", "switches")),
            *(1.0 if vd.get("category") == c else 0.0 for c in CATS), vf]


def build_features(feature_set: str, scene_ratio: float, dance_ratio: float, explicit_ratio: float,
                   frames_stats: dict | None, video: dict | None, frames3_stats: dict | None = None,
                   audio: list[float] | None = None) -> list[float]:
    x = features_v2(scene_ratio, dance_ratio, explicit_ratio, frames_stats, video)
    if "frames_v3" in feature_set:  # 備援版：場景動作、遮蔽畫面的比例
        f3 = frames3_stats or {}
        x += [float(f3.get(k, 0.0)) for k in ("D", "S", "O", "switches", "p", "z")]
    if "audio" in feature_set:  # 聲音：節拍、人聲、音色（見 vsort/audio.py）
        from vsort.audio import FEATURE_NAMES
        au = list(audio or [])
        x += au if len(au) == len(FEATURE_NAMES) else [0.0] * len(FEATURE_NAMES)
    return x


def feature_names(feature_set: str) -> list[str]:
    names = list(FEATURES)
    if "frames_v3" in feature_set:
        names += ["備援版：跳舞", "備援版：場景", "備援版：其他", "備援版：切換", "備援版：場景動作", "備援版：遮蔽"]
    if "audio" in feature_set:
        from vsort.audio import FEATURE_NAMES
        names += [f"聲音：{n}" for n in FEATURE_NAMES]
    return names


def features(scene_ratio: float, dance_ratio: float, explicit_ratio: float, qwen_json: dict | None,
             qwen_frames: int = 8) -> list[float]:
    """舊版特徵（整部描述的 JSON），保留給比較工具用。"""
    q = (qwen_json or {}).get("category")
    sf = (qwen_json or {}).get("scene_frames")
    try:
        sf = min(float(sf) / qwen_frames, 1.0)
    except (TypeError, ValueError):
        sf = 0.5
    return [scene_ratio, dance_ratio, explicit_ratio, *(1.0 if q == c else 0.0 for c in CATS), sf]


def _dataset(cfg, analyzer, progress=print, feature_set: str = FEATURE_SET):
    from vsort.labeling import load_labels
    items = [v for v in load_labels(cfg).values() if v.get("category") in CATS and Path(v["path"]).exists()]
    X, y = [], []
    for i, v in enumerate(items, 1):
        p = Path(v["path"])
        ai = analyzer.run(p)
        fs, vd, f3 = analyzer.qwen_signals(p, fallback="frames_v3" in feature_set)
        au = analyzer.audio_features(p) if "audio" in feature_set else None
        X.append(build_features(feature_set, ai.scene_ratio, ai.dance_ratio, ai.explicit_ratio, fs, vd, f3, au))
        y.append(CATS.index(v["category"]))
        if i % 25 == 0:
            progress(f"  {i}/{len(items)}")
    return np.array(X), np.array(y)


def evaluate_and_train(cfg, analyzer, progress=print, sure_prob: float = 0.8,
                       feature_set: str = FEATURE_SET, mix_weight: float = MIX_WEIGHT,
                       out_path: Path = CATEGORY_CLASSIFIER) -> dict:
    X, y = _dataset(cfg, analyzer, progress, feature_set)
    return fit(X, y, sure_prob, feature_set, mix_weight, out_path)


def fit(X, y, sure_prob: float = 0.8, feature_set: str = FEATURE_SET, mix_weight: float = MIX_WEIGHT,
        out_path: Path = CATEGORY_CLASSIFIER) -> dict:
    """交叉驗證挑正則化強度，再用全部樣本訓練並存檔。含聲音特徵時先標準化（尺度差很多）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y) < 20 or len(set(y)) < 2:
        raise RuntimeError(f"有類別答案的影片只有 {len(y)} 部，至少需要 20 部、而且要有兩種以上類別。")
    cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=0)
    cw = {0: 1.0, 1: mix_weight, 2: 1.0}
    best = None
    scaled = "audio" in feature_set

    def model(C):
        lr = LogisticRegression(C=C, max_iter=3000, class_weight=cw)
        return make_pipeline(StandardScaler(), lr) if scaled else lr
    for C in ((0.03, 0.1, 0.3, 1.0, 3.0) if scaled else (0.3, 1.0, 3.0)):
        p = cross_val_predict(model(C), X, y, cv=cv, method="predict_proba")
        acc = float((p.argmax(1) == y).mean())
        if best is None or acc > best[0]:
            best = (acc, C, p)
    acc, C, p = best
    conf = p.max(1) >= sure_prob
    qwen_only = float((X[:, 7:10].argmax(1) == y)[X[:, 7:10].sum(1) > 0].mean())  # 影片模式單獨
    report = {"n": int(len(y)), "counts": {c: int((y == i).sum()) for i, c in enumerate(CATS)},
              "accuracy": round(acc, 3), "qwen_only": round(qwen_only, 3), "C": C,
              "sure_share": round(float(conf.mean()), 3),
              "accuracy_when_sure": round(float((p.argmax(1) == y)[conf].mean()), 3) if conf.any() else None,
              "per_class_recall": {c: round(float((p.argmax(1)[y == i] == i).mean()), 3) for i, c in enumerate(CATS)}}
    m = model(C).fit(X, y)
    lr = m[-1] if scaled else m
    d = {"classes": [CATS[c] for c in lr.classes_], "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
         "features": feature_names(feature_set), "feature_set": feature_set, "report": report}
    if scaled:
        d["mean"], d["scale"] = m[0].mean_.tolist(), m[0].scale_.tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return report


class CategoryClassifier:
    def __init__(self, path: Path = CATEGORY_CLASSIFIER):
        d = json.loads(path.read_text(encoding="utf-8"))
        self.feature_set = d.get("feature_set", "tags+qwen_json")
        self.classes = d["classes"]
        self.coef, self.intercept = np.array(d["coef"]), np.array(d["intercept"])
        self.mean = np.array(d["mean"]) if "mean" in d else None
        self.scale = np.array(d["scale"]) if "scale" in d else None

    def predict(self, x: list[float]) -> dict[str, float]:
        x = np.array(x, dtype=float)
        if self.mean is not None:
            x = (x - self.mean) / self.scale
        z = self.coef @ x + self.intercept
        e = np.exp(z - z.max())
        return dict(zip(self.classes, (e / e.sum()).tolist()))
