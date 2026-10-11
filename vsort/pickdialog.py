"""跳出 Windows 原生的選檔／選資料夾視窗，把選到的路徑印成 JSON。

獨立成一個小程式（python -m vsort.pickdialog files|folder）由網頁伺服器用子行程呼叫：
Tkinter 不能在伺服器的工作執行緒裡開視窗，用子行程最穩。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def pick(kind: str) -> list[str]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "folder":
            chosen = filedialog.askdirectory(title="選擇要分類的影片資料夾（裡面的影片全部）", mustexist=True)
            paths = [chosen] if chosen else []
        else:
            paths = list(filedialog.askopenfilenames(
                title="選擇要分類的影片（可複選）",
                filetypes=[("影片", "*.mp4 *.mkv *.webm *.mov *.avi *.wmv *.flv *.m4v *.ts *.mpg *.mpeg *.3gp"), ("全部", "*.*")]))
    finally:
        root.destroy()
    return [str(Path(p)) for p in paths]


def pick_in_subprocess(kind: str, timeout: int = 600) -> list[str]:
    """由另一個程序開視窗（網頁伺服器用）。使用者按取消就回傳空清單。"""
    out = subprocess.run([sys.executable, "-m", "vsort.pickdialog", kind], capture_output=True, timeout=timeout,
                         creationflags=_NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace").strip().splitlines()[-1:] or "選擇視窗開不起來")
    return json.loads(out.stdout.decode("utf-8") or "[]")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "files"
    sys.stdout.buffer.write(json.dumps(pick(kind), ensure_ascii=False).encode("utf-8"))
