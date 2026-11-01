"""文本大模型（LLM）封装：System Prompt、RAG 上下文拼装与流式生成。

TextLLM 是上层对话的入口：把用户问题（可选历史、可选知识库检索结果）
组织成 chat 消息发给 llama-server 上的 Qwen3.5-4B，支持一次性与流式两种用法。
被 AssistantService / RAGPipeline 调用；模型进程的管理在 src/llama_backend.py。
"""
from __future__ import annotations

SYSTEM_PROMPT = """你是多模态AI助手。你通过语音、文本、摄像头画面和本地知识库帮助用户。
回答应友好、专业、准确，优先使用简洁自然的中文。不要声称看到了未提供的画面，也不要编造知识库内容。
回答使用适合聊天显示和语音朗读的普通文本，不使用 Markdown 标题、加粗、星号列表、表格或代码围栏；分点时用自然语言分行说明。
不确定时明确说明不确定，并给出可验证的下一步。"""


class TextLLM:
    """本地 llama.cpp（llama-server）文本模型客户端（token 级流式）。

    公开接口与早期 Ollama 版本保持一致；base_url / keep_alive / num_ctx
    参数保留用于向后兼容，运行时实际连接由 config.LLAMA_CONFIG
    （Qwen3.5-4B IQ4_XS，llama-server 提供）或注入的 client 决定。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 300,
        keep_alive: str = "10m",
        num_ctx: int = 8192,
        client=None,
    ):
        self.model_name = model_name
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        if client is None:
            from .llama_backend import create_text_client

            client = create_text_client()
        self.client = client

    def health(self) -> dict:
        """列出可用模型名，判断本客户端要用的模型是否已就绪。"""
        models = self.client.list_models()
        return {
            "ok": self.model_name in models,
            "model": self.model_name,
            "available_models": models,
        }

    def _build_messages(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """拼装 chat 消息：System Prompt（+知识库上下文）→ 历史 → 当前问题。"""
        system = SYSTEM_PROMPT
        if kb_mode:
            context = kb_context.strip() or "（当前没有检索到相关资料）"
            system += (
                "\n\n回答时优先依据下面的本地参考资料。资料不足时先说明资料不足，"
                "再明确区分地使用通用知识补充。\n【参考资料】\n" + context
            )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-12:])  # 只带最近 12 条，控制上下文长度
        messages.append({"role": "user", "content": user_text})
        return messages

    def _build_payload(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """构造发往 llama-server 的请求体（温度 0.4、指定上下文长度）。"""
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(user_text, kb_context, kb_mode, history),
            "keep_alive": self.keep_alive,
            "think": False,
            "options": {"temperature": 0.4, "num_ctx": self.num_ctx},
        }
        if max_new_tokens:
            payload["options"]["num_predict"] = max_new_tokens
        return payload

    def generate(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> str:
        """一次性生成：发给后端并返回完整回答文本。"""
        payload = self._build_payload(user_text, kb_context, kb_mode, max_new_tokens, history)
        response = self.client.chat(payload)
        return str(response.get("message", {}).get("content", "")).strip()

    def stream_generate(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ):
        """流式生成：逐段 yield 内容片段（界面「边生成边显示」用）。"""
        payload = self._build_payload(user_text, kb_context, kb_mode, max_new_tokens, history)
        for event in self.client.stream_chat(payload):
            content = event.get("message", {}).get("content", "")
            if content:
                yield content
