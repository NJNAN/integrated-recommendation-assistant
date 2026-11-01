"""模态路由：根据用户输入判断走「文本」还是「视觉」链路。

优先识别 /vision、/text 手动命令；自动模式下命中视觉关键词即切到视觉。
（历史上的单字「看」已移除，避免「看完」之类的日常用词误触发模式切换。）
每次切换会记录带时间戳的历史，便于排查模式抖动。
"""
from __future__ import annotations

from datetime import datetime


class ModalRouter:
    """文本 / 视觉模式路由（关键词 + 手动命令）。"""

    # 视觉意图关键词（中英文）
    VISION_KEYWORDS = [
        "画面",
        "摄像头",
        "镜头",
        "屏幕",
        "图像",
        "图片",
        "照片",
        "看看",
        "看一下",
        "看下",
        "看一眼",
        "看这",
        "看那",
        "看到",
        "看见",
        "瞅瞅",
        "拍照",
        "拍一下",
        "拍到",
        "拍下",
        "拍张",
        "拍个",
        "look",
        "see",
        "camera",
        "view",
        "screenshot",
        "photo",
    ]

    def __init__(self):
        self.current_mode = "text"
        self.auto_mode = True
        self.history: list[dict] = []

    def route(self, user_input: str) -> str:
        """给一条用户输入做路由，返回当前模式（"text" / "vision"）。"""
        text = user_input.strip().lower()
        if text.startswith("/vision"):
            self.set_mode("vision", auto=False)
        elif text.startswith("/text"):
            self.set_mode("text", auto=False)
        elif self.auto_mode:
            mode = "vision" if any(key in text for key in self.VISION_KEYWORDS) else "text"
            self.set_mode(mode, auto=True)
        return self.current_mode

    def set_mode(self, mode: str, auto: bool | None = None) -> None:
        """切换模式（auto 非 None 时同时更新自动开关）；模式变化时记历史。"""
        if mode not in {"text", "vision"}:
            raise ValueError("mode 必须是 text 或 vision")
        changed = mode != self.current_mode
        self.current_mode = mode
        if auto is not None:
            self.auto_mode = auto
        if changed:
            self.history.append(
                {"timestamp": datetime.now().isoformat(timespec="seconds"), "mode": mode}
            )

    def enable_auto(self) -> None:
        """重新打开自动路由（手动命令后可恢复）。"""
        self.auto_mode = True
