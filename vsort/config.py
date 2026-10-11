"""設定檔讀取（F14）與資料夾配置（F01、F30）。"""

from __future__ import annotations

import re

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULTS: dict = {
    "workspace_dir": "Workspace",
    "library_dir": None,
    "dictionary_dir": "dict",
    "parser": {},
    "folders": {
        "inbox": "01_Inbox",
        "staging": "02_Parsed_Staging",
        "ai": "03_AI_Workspace",
        "identify": "Identify_Queue",
        "refine": "Refine_Queue",
        "library": "04_Final_Library",
        "review": "05_Review",
        "logs": "_logs",
        "reports": "_reports",
    },
    "video_extensions": [".mp4", ".mkv", ".webm", ".mov", ".avi", ".wmv", ".flv", ".m4v", ".ts"],
    "temp_extensions": [".part", ".crdownload", ".tmp", ".download", ".aria2"],
    "min_file_age_seconds": 60,
    "naming": {
        "max_name_length": 180,
        "max_path_length": 250,
        "illegal_char_style": "fullwidth",
    },
    "ingest": {
        "mode": "move",
        "keep_folder_structure": True,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        elif value is not None or key not in out:
            out[key] = value
    return out


@dataclass
class Layout:
    """工作區各資料夾的實際路徑。"""

    workspace: Path
    inbox: Path
    staging: Path
    ai: Path
    identify: Path
    refine: Path
    library: Path
    review: Path
    logs: Path
    reports: Path

    def stage_dirs(self) -> list[tuple[str, Path]]:
        return [
            ("01 原始輸入", self.inbox),
            ("02 解析分流", self.staging),
            ("03 AI 待辨識", self.identify),
            ("03 AI 待補標", self.refine),
            ("04 收藏庫", self.library),
            ("05 人工確認", self.review),
        ]

    def ensure(self) -> list[Path]:
        """建立所有資料夾，回傳新建立的。"""
        created = []
        for path in [self.workspace, self.inbox, self.staging, self.ai, self.identify,
                     self.refine, self.library, self.review, self.logs, self.reports]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(path)
        return created


@dataclass
class Config:
    data: dict
    path: Path | None
    base_dir: Path
    layout: Layout = field(init=False)

    def __post_init__(self) -> None:
        f = self.data["folders"]
        root = self.data.get("root_dir")
        if root:  # 一個根目錄：底下 sorting\01～05 是工作中的，作者資料夾直接放根目錄
            ws = self._resolve(root) / (f.get("sorting") or "sorting")
            lib = self.data.get("library_dir") or root
        else:
            ws = self._resolve(self.data["workspace_dir"])
            lib = self.data.get("library_dir")
        ai = ws / f["ai"]
        self.layout = Layout(
            workspace=ws,
            inbox=ws / f["inbox"],
            staging=ws / f["staging"],
            ai=ai,
            identify=ai / f["identify"],
            refine=ai / f["refine"],
            library=self._resolve(lib) if lib else ws / f["library"],
            review=ws / f["review"],
            logs=ws / f["logs"],
            reports=ws / f["reports"],
        )

    def _resolve(self, value: str | Path) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.base_dir / p
        return p.resolve()

    @property
    def root_dir(self) -> Path | None:
        r = self.data.get("root_dir")
        return self._resolve(r) if r else None

    @property
    def video_extensions(self) -> set[str]:
        return {e.lower() for e in self.data["video_extensions"]}

    @property
    def skip_folders(self) -> set[str]:
        """抽樣與重測時整個略過的資料夾名稱（不分大小寫）。"""
        return {str(x).lower() for x in (self.data.get("skip_folders") or [])}

    def skipped(self, path, root=None) -> bool:
        parts = path.relative_to(root).parts if root is not None else path.parts
        return any(p.lower() in self.skip_folders for p in parts[:-1])

    @property
    def temp_extensions(self) -> set[str]:
        return {e.lower() for e in self.data["temp_extensions"]}

    @property
    def dictionary_dir(self) -> Path:
        return self._resolve(self.data["dictionary_dir"])

    @property
    def naming(self) -> dict:
        return self.data["naming"]


def set_root_dir(cfg_path: Path, root: Path) -> None:
    """把分類根目錄寫進設定檔（只改 root_dir 這一行，其他註解與設定保留）。"""
    text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
    line = f"root_dir: '{root}'"
    if re.search(r"^root_dir:.*$", text, flags=re.M):
        text = re.sub(r"^root_dir:.*$", lambda _m: line, text, count=1, flags=re.M)  # 路徑含反斜線，不能當樣板
    else:
        text = ("# 分類根目錄：底下 sorting\\01～05 是工作中的影片，作者資料夾直接放這裡（設定後 workspace_dir、library_dir 不用管）\n"
                + line + "\n\n" + text)
    cfg_path.write_text(text, encoding="utf-8", newline="")


def load_config(path: str | Path | None = None, library_override: str | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user: dict = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    elif path:
        raise FileNotFoundError(f"找不到設定檔：{cfg_path}")
    data = _merge(DEFAULTS, user)
    if library_override:
        data["library_dir"] = library_override
    base = cfg_path.resolve().parent if cfg_path.exists() else PROJECT_ROOT
    return Config(data=data, path=cfg_path if cfg_path.exists() else None, base_dir=base)
