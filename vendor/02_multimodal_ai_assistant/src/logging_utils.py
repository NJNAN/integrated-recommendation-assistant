"""日志初始化：同时输出到文件与终端，格式统一，只配置一次。"""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path, level: str = "INFO") -> None:
    """配置根日志：文件 + 控制台双通道；重复调用不会叠加 handler。"""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return  # 已配置过：直接跳过，避免重复输出
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
