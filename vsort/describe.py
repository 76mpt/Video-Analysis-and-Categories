"""看畫面寫描述的大模型（選用）：JoyCaption、Qwen3-VL。

用途：給「可能誤判」的影片多一份意見，並在人工確認時顯示文字描述。
兩個模型都用 4-bit 載入（bitsandbytes），16 GB 顯卡放得下；一次只載一個。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vsort.tagger import MODELS_DIR  # noqa: F401  sets HF_HOME to models/

QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"
JOY_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"

# Prompts live in prompts/: a local <name>.txt (git-ignored) overrides the shipped <name>.example.txt.
# The category definitions (Dance / Mix / Scene) are part of the prompts, so the classification
# criteria can be changed without touching the code.
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    for fn in (f"{name}.txt", f"{name}.example.txt"):
        f = PROMPT_DIR / fn
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(PROMPT_DIR / f"{name}.example.txt")


PROMPT_VERSION = 2  # 改了提示詞就加一，快取會自動失效
PROMPT = load_prompt("describe")
JOY_PROMPT = load_prompt("joycaption")


def _bnb():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=["vision_tower", "visual", "multi_modal_projector", "lm_head"])


def parse_json_tail(text: str) -> dict:
    """從模型回答裡抓最後一段 JSON；animation、characters 一律整理成清單。"""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        for k in ("animation", "characters"):
            v = d.get(k)
            d[k] = [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
        return d
    return {}


def _resize(img, side: int):
    im = img.copy()
    im.thumbnail((side, side))
    return im


class QwenDescriber:
    name = "qwen3vl"

    def __init__(self, repo: str = QWEN_REPO):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo, quantization_config=_bnb(), dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(repo)

    def describe(self, images: list, max_frames: int = 8) -> dict:
        step = max(1, len(images) // max_frames)
        imgs = [_resize(im, 640) for im in images[::step][:max_frames]]
        msgs = [{"role": "user", "content": [*({"type": "image"} for _ in imgs),
                                             {"type": "text", "text": PROMPT.format(n=len(imgs))}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        answer = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return {"model": self.name, "text": answer, "json": parse_json_tail(answer)}


class JoyDescriber:
    name = "joycaption"

    def __init__(self, repo: str = JOY_REPO):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo, quantization_config=_bnb(), dtype=torch.bfloat16, device_map="cuda").eval()

    def describe(self, images: list, max_frames: int = 9) -> dict:
        from vsort.frames import contact_sheet
        step = max(1, len(images) // max_frames)
        picked = images[::step][:max_frames]
        sheet = contact_sheet([(0, im) for im in picked], cols=3, cell=384)
        convo = [{"role": "system", "content": "You are a helpful image captioner."},
                 {"role": "user", "content": JOY_PROMPT.format(n=len(picked))}]
        text = self.proc.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[sheet], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.bfloat16)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=350, do_sample=False)
        answer = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return {"model": self.name, "text": answer, "json": {}}


DESCRIBERS = {"qwen3vl": QwenDescriber, "joycaption": JoyDescriber}


FRAMES_PROMPT_VERSION = 1  # v1 is in use; v2 / v3 (extra yes/no flags per frame) are kept for comparison
FRAMES_PROMPT_V1 = load_prompt("frames_v1")
FRAMES_PROMPT_V2 = load_prompt("frames_v2")
FRAMES_PROMPT_V3 = load_prompt("frames_v3")
FRAMES_PROMPTS = {1: FRAMES_PROMPT_V1, 2: FRAMES_PROMPT_V2, 3: FRAMES_PROMPT_V3}


def frame_category(letters: list) -> tuple[str | None, dict]:
    """Per-frame labels -> whole-video category: >= 70% S frames is Scene, <= 1 S frame is Dance, else Mix.
    v2 / v3 frames are {"c": "D/S/O"} plus 0/1 flags (g, m, p, z); any flag set counts the frame as S."""
    seq, g_cnt, m_cnt, p_cnt, z_cnt = [], 0, 0, 0, 0
    for v in letters:
        if isinstance(v, dict):
            g, m = _flag(v.get("g")), _flag(v.get("m"))
            p, z = _flag(v.get("p")), _flag(v.get("z"))
            g_cnt, m_cnt, p_cnt, z_cnt = g_cnt + g, m_cnt + m, p_cnt + p, z_cnt + z
            c = str(v.get("c") or "").strip().upper()[:1]
            seq.append("S" if (g or m or p or z) else c)
        else:
            seq.append(str(v).strip().upper()[:1])
    seq = [x for x in seq if x in "DSO"]
    if not seq:
        return None, {}
    n = len(seq)
    stats = {"D": seq.count("D") / n, "S": seq.count("S") / n, "O": seq.count("O") / n,
             "switches": sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {"D", "S"}) / max(n - 1, 1),
             "g": g_cnt / n, "m": m_cnt / n, "p": p_cnt / n, "z": z_cnt / n}
    s = seq.count("S")
    cat = "Scene" if s / n >= 0.7 else "Dance" if s <= 1 else "Mix"
    return cat, stats


def _flag(v) -> int:
    try:
        return 1 if int(v) else 0
    except (TypeError, ValueError):
        return 1 if str(v).strip().lower() in ("yes", "true", "y") else 0


def _qwen_classify_frames(self, images: list, max_frames: int = 16, version: int = FRAMES_PROMPT_VERSION) -> dict:
    step = max(1, len(images) // max_frames)
    imgs = [_resize(im, 512) for im in images[::step][:max_frames]]
    content = []
    for i, _ in enumerate(imgs, 1):
        content += [{"type": "text", "text": f"Frame {i}:"}, {"type": "image"}]
    content.append({"type": "text", "text": FRAMES_PROMPTS[version].format(n=len(imgs))})
    text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    inputs = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
    with self.torch.inference_mode():
        out = self.model.generate(**inputs, max_new_tokens=160 if version == 1 else 420, do_sample=False)
    answer = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    m = re.search(r"\{.*\}", answer, re.S)
    letters = []
    if m:
        try:
            letters = json.loads(m.group(0)).get("frames") or []
        except ValueError:
            if version == 1:
                letters = re.findall(r'"([DSO])"', m.group(0))
            else:  # JSON 不完整時，逐筆抓 {"g":..,"m":..,"c":..}
                letters = [json.loads(x) for x in re.findall(r'\{[^{}]*"c"\s*:\s*"[DSO]"[^{}]*\}', m.group(0))]
    cat, stats = frame_category(letters)
    return {"model": self.name, "text": answer, "frames": letters, "category": cat, "stats": stats}


QwenDescriber.classify_frames = _qwen_classify_frames


VIDEO_PROMPT = load_prompt("video")


def _qwen_classify_video(self, images: list, max_frames: int = 16, fps: float = 1.0) -> dict:
    """影片模式：把關鍵幀當成一段影片給 Qwen3-VL，讓它知道先後順序。"""
    step = max(1, len(images) // max_frames)
    frames = [_resize(im, 448) for im in images[::step][:max_frames]]
    if len(frames) % 2:  # Qwen3-VL 的影片要偶數張
        frames = frames[:-1] or frames * 2
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": VIDEO_PROMPT}]}]
    text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = self.proc(text=[text], videos=[frames], return_tensors="pt", do_sample_frames=False,
                       video_metadata=[{"fps": fps, "total_num_frames": len(frames),
                                                       "frames_indices": list(range(len(frames)))}]).to(self.model.device)
    with self.torch.inference_mode():
        out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    m = re.search(r"\{.*\}", answer, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = {}
    cat = d.get("category") if d.get("category") in ("Dance", "Mix", "Scene") else None
    try:
        frac = float(d.get("scene_fraction"))
    except (TypeError, ValueError):
        frac = None
    return {"model": self.name, "text": answer, "category": cat, "scene_fraction": frac}


QwenDescriber.classify_video = _qwen_classify_video
