"""手势 → 动作 映射模块。

本模块把 ``GestureRecognizer`` 输出的六类手势翻译成具体的应用层动作（音
量键、截图、停止语音播报、静音、开始监听），并按是否传入 ``callbacks``
决定是调用回调还是直接走系统级实现：

- ``volume_up`` / ``volume_down`` / ``mute``：通过 ``ctypes.windll.user32``
  模拟 Windows 媒体键（``VK_VOLUME_UP/DOWN/MUTE``），无需 pycaw；
- ``stop``：停掉当前 TTS 播放器（``tts_player.stop()``）；
- ``start_listening`` / ``take_snapshot``：默认无系统实现，但前者通常通过
  回调把状态交给上层 UI，后者由 ``frame`` + ``output_path`` 写入图片。
- 如果调用方在 ``kwargs`` 里传了 ``callbacks``（GUI / 仿真测试环境），会
  优先执行回调，便于在没有音频 / 摄像头子系统的环境下做行为验证。

注意：``_press_media_key`` 走 Windows 原生 API，仅 Windows 可用；``take_snapshot``
会延迟 ``import`` ``image_utils``，因为大多数动作路径不需要它。
"""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class GestureActionMapper:
    """把手势标签映射到应用动作并执行。

    Attributes:
        action_map: 手势标签 → 动作名称 的硬编码字典。
    """

    def __init__(self):
        """初始化手势→动作映射表。"""
        self.action_map = {
            "thumbs_up": "volume_up",
            "thumbs_down": "volume_down",
            "open_palm": "stop",
            "index_up": "start_listening",
            "index_down": "mute",
            "victory": "take_snapshot",
        }

    def map(self, gesture: str) -> str:
        """把手势标签翻译成动作名；未识别的手势返回 ``"unknown"``。"""
        return self.action_map.get(gesture, "unknown")

    def execute(self, action: str, **kwargs):
        """执行 ``action`` 名称对应的实现。

        Args:
            action: ``map`` 返回的动作名。
            **kwargs: 透传给具体实现的上下文。常用键：
                - ``callbacks``：dict[action_name, callable]，若有则优先调
                  用回调（用于无音频 / 测试环境）；
                - ``tts_player``：``stop`` 动作需要；
                - ``frame`` + ``output_path``：``take_snapshot`` 动作需要。

        Returns:
            ``True`` 表示执行成功（含回调命中）；``False`` 表示无可执行目
            标或底层失败。
        """
        callbacks = kwargs.get("callbacks", {})
        # 优先走调用方注入的回调：GUI / 测试 / 仿真环境下不需要碰真硬件
        if action in callbacks:
            return callbacks[action]()
        if action == "volume_up":
            return self._press_media_key(0xAF)
        if action == "volume_down":
            return self._press_media_key(0xAE)
        if action == "mute":
            return self._press_media_key(0xAD)
        if action == "stop":
            player = kwargs.get("tts_player")
            if player:
                player.stop()
                return True
        if action == "take_snapshot":
            frame = kwargs.get("frame")
            output_path = kwargs.get("output_path")
            if frame is not None and output_path:
                # image_utils 只在截图路径才需要，延迟导入以减少冷启动开销
                from .image_utils import write_image

                path = Path(output_path)
                write_image(path, frame)
                return True
        # 没有可执行目标（含未知手势映射到 "unknown"）只记日志，不抛异常
        LOGGER.info("动作 %s 没有可执行目标", action)
        return False

    @staticmethod
    def _press_media_key(code: int) -> bool:
        """通过 ctypes 模拟 Windows 媒体键（仅 Windows）。

        Args:
            code: ``VK_VOLUME_UP``(0xAF) / ``VK_VOLUME_DOWN``(0xAE) /
                ``VK_VOLUME_MUTE``(0xAD) 之一。
        """
        try:
            user32 = ctypes.windll.user32
            # 0 = 按下，2 = 抬起；中间的 scan code 等参数对媒体键无影响
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, 2, 0)
            return True
        except (AttributeError, OSError):
            # 非 Windows / user32 不可用 → 静默失败，不影响其它动作
            return False