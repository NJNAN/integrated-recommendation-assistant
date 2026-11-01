"""Ollama HTTP 客户端（遗留组件；当前主链路的推理已改用 llama.cpp）。

保留用于对照与回退：早期版本的文本 / 视觉模型由本机 Ollama 服务承载，
本客户端的接口语义（list_models / show_model / chat / stream_chat）
与 src/llama_backend.py 中的 llama.cpp 版本保持一致，便于相互对照阅读。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaClient:
    """Ollama REST 客户端（对照 llama_backend.LlamaCppClient 阅读）。"""

    def __init__(self, base_url: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # 本机模型流量不允许走系统代理
        # Localhost model traffic must never be routed through a system proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, payload: dict | None = None):
        """构造并发送请求；连接失败时给出「请确认 Ollama 已启动」的可读报错。"""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 Ollama ({self.base_url})，请确认 Ollama 已启动"
            ) from exc

    def list_models(self) -> list[str]:
        """GET /api/tags：返回本机已安装的模型名列表。"""
        with self._request("/api/tags") as response:
            payload = json.load(response)
        return [item.get("name", "") for item in payload.get("models", [])]

    def show_model(self, name: str) -> dict:
        """POST /api/show：查看单个模型的详细信息（含能力列表）。"""
        with self._request("/api/show", {"model": name}) as response:
            return json.load(response)

    def chat(self, payload: dict) -> dict:
        """POST /api/chat（stream=False）：一次性返回完整回答。"""
        payload = {**payload, "stream": False}
        with self._request("/api/chat", payload) as response:
            return json.load(response)

    def stream_chat(self, payload: dict):
        """POST /api/chat（stream=True）：逐行 JSON 流式返回增量。"""
        payload = {**payload, "stream": True}
        with self._request("/api/chat", payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
