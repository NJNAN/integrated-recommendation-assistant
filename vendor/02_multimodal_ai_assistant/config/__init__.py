"""配置包入口：统一从这里导出 config/settings.py 中的全部配置项。

项目的其它模块一律通过 `from config import ...` 获取配置（如 PATHS、MODEL_CONFIG），
不要绕过本文件直接 import config.settings —— 保证「配置只有一个来源」。
"""

from .settings import (
    AUDIO_CONFIG,
    DEVICE,
    LLAMA_CONFIG,
    LOG_CONFIG,
    MODEL_CONFIG,
    PATHS,
    PROJECT_ROOT,
    SEED,
    TTS_CONFIG,
    VIDEO_CONFIG,
    set_seed,
)

# 显式声明本包对外暴露的名字（供 `from config import *` 与静态检查使用）
__all__ = [
    "PROJECT_ROOT",
    "DEVICE",
    "SEED",
    "PATHS",
    "MODEL_CONFIG",
    "LLAMA_CONFIG",
    "AUDIO_CONFIG",
    "VIDEO_CONFIG",
    "TTS_CONFIG",
    "LOG_CONFIG",
    "set_seed",
]
