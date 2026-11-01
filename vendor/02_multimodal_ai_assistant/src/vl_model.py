"""视觉语言模型适配层：把摄像头帧 / 图片交给 llama-server 上的 Qwen3-VL 推理。

对外只暴露 VLModel 一个类：把「最近两帧」编码为 base64 JPEG，
按与文本模型一致的 chat 协议调用 llama.cpp 后端，返回回答文本。
由 AssistantService / 模态路由调用。
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class VLModel:
    """Qwen3-VL 的本地 llama.cpp（llama-server）视觉适配器。

    模型与其视觉投影层（mmproj）由 llama-server 在首次使用时一起加载。
    按课程要求，只发送「最新的两帧」作为时序上下文。
    """

    MAX_FRAMES = 2  # 只送最近两帧：2 帧足够看清主体，prefill 减半

    def __init__(
        self,
        model_name: str = "qwen3-vl-4b-instruct",
        base_url: str = "http://127.0.0.1:8081",
        timeout: float = 300,
        keep_alive: str = "5m",
        client=None,
    ):
        self.model_name = model_name
        self.keep_alive = keep_alive
        if client is None:
            from .llama_backend import create_vision_client

            client = create_vision_client()
        self.client = client

    def health(self) -> dict:
        """检查视觉模型可用性：模型在列且具备图像输入能力。"""
        models = self.client.list_models()
        if self.model_name not in models:
            return {"ok": False, "reason": f"模型不可用 {self.model_name}", "models": models}
        details = self.client.show_model(self.model_name)
        capabilities = details.get("capabilities", [])
        has_vision = "vision" in capabilities
        return {
            "ok": has_vision,
            "reason": "" if has_vision else "视觉模型缺少图像输入能力",
            "capabilities": capabilities,
        }

    def create_messages(self, video_paths: list[str], text_prompt: str) -> list[dict]:
        """构造带图片的消息列表（文件路径版，供脚本直接测试）。"""
        selected = video_paths[-self.MAX_FRAMES :]
        return [
            {
                "role": "system",
                "content": "你是多模态AI助手的视觉模块。只根据提供的画面回答，无法确认时明确说明。",
            },
            {
                "role": "user",
                "content": text_prompt,
                "images": [self._encode_path(path) for path in selected],
            },
        ]

    def generate_response(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        """把消息发给后端并返回回答文本（低温 0.2，限制生成长度）。"""
        response = self.client.chat(
            {
                "model": self.model_name,
                "messages": messages,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0.2, "num_predict": max_new_tokens},
            }
        )
        return str(response.get("message", {}).get("content", "")).strip()

    def analyze_frames(
        self,
        frames: list,
        text_prompt: str,
        max_new_tokens: int = 256,
        history: list[dict] | None = None,
    ) -> str:
        """主入口：分析「最近两帧」（ndarray 或路径），可携带历史做多轮追问。"""
        selected = frames[-self.MAX_FRAMES :]
        if not selected:
            raise ValueError("没有可供视觉理解的摄像头帧")
        images = [self._encode_frame(frame) for frame in selected]
        messages = [
            {
                "role": "system",
                "content": "你是多模态AI助手的视觉模块。只根据提供的画面回答，无法确认时明确说明。",
            }
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text_prompt, "images": images})
        return self.generate_response(messages, max_new_tokens)

    @staticmethod
    def _encode_path(path: str | Path) -> str:
        """读文件并编码为 base64（llama-server 接受的图片传参格式）。"""
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")

    @staticmethod
    def _encode_frame(frame) -> str:
        """把一帧（路径或 OpenCV ndarray）编码成 base64 JPEG（质量 88）。"""
        if isinstance(frame, (str, Path)):
            return VLModel._encode_path(frame)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python") from exc
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not ok:
            raise ValueError("摄像头帧 JPEG 编码失败")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
