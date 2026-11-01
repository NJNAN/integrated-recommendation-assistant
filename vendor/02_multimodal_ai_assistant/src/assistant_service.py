"""助手服务层：串行化用户请求，统一文本 / 视觉两条回答路径。

AssistantService 是界面与模型之间的“指挥”：
  - 用锁保证同一时刻只处理一个请求（新请求排队等待）；
  - 文本路径：检索知识库 → 流式生成 → 逐段回调；
  - 视觉路径：取最近两帧 / 上传图片 → 视觉观察 → 检索资料 → 流式融合；
  - 知识库无命中时保留视觉直接回答；会话记忆在“新对话”时清空；
  - 被取消的回答不进入对话历史，也不产生推理指标。
推理细节（耗时、首段延迟、token、来源）经 ResponseDetails 回传给界面。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .session_memory import SessionMemory
from .text_presentation import to_plain_text

LOGGER = logging.getLogger(__name__)


@dataclass
class ResponseDetails:
    """一次回答的完整推理细节（导出对话 / 界面展示用）。"""

    mode: str  # text / vision
    elapsed_seconds: float  # 端到端耗时
    first_chunk_seconds: float | None  # 首段输出延迟（未输出时为 None）
    cancelled: bool = False
    sources: list[dict] = field(default_factory=list)  # 检索到的资料块
    inference: dict = field(default_factory=dict)  # token / TPOT 等原生指标


class AssistantService:
    """把「问题 → 回答」的完整流程封装成可取消、可观测的服务。"""

    def __init__(self, text_llm, rag, vl_model, router, video_capture):
        self.text_llm = text_llm
        self.rag = rag
        self.vl_model = vl_model
        self.router = router
        self.video_capture = video_capture
        self.history: list[dict] = []
        self._memory = SessionMemory()
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()  # 保护 cancel_event 的替换
        self._process_lock = threading.Lock()  # 请求串行化

    def cancel(self) -> None:
        """取消当前处理中的回答（正在发送的分片会被丢弃）。"""
        with self._state_lock:
            self._cancel_event.set()

    def reset_history(self) -> None:
        """清空对话历史（“新对话”按钮）。"""
        with self._process_lock:
            self.history.clear()
            self._memory.clear()
            LOGGER.info("会话记忆已清空")

    @property
    def memory_summary(self) -> str:
        """供界面展示当前会话的简短记忆，不含图片或跨会话持久化。"""
        return self._memory.summary

    @staticmethod
    def _clean_query(query: str) -> str:
        """去掉开头的 /vision、/text 命令词，保留消息其余部分。"""
        # Preserve literal commands in the middle of a user's message.
        for command in ("/vision", "/text"):
            if query.lower().startswith(command):
                return query[len(command):].strip()
        return query.strip()

    def process(
        self, query: str, on_chunk=None, *, frames=None, cancel_event=None,
        mode: str | None = None, on_details=None,
    ) -> str:
        """处理一次请求并返回回答文本（核心入口）。

        请求串行化；被取消的回答不写入对话历史。
        ``frames`` 参数可直接传入本地上传的图片（无需摄像头）；
        ``cancel_event`` 允许把「处理开始前就收到停止」的情况也纳入。
        兼容旧签名 ``process(query, on_chunk)``。
        """
        with self._process_lock:
            event = cancel_event if cancel_event is not None else threading.Event()
            with self._state_lock:
                self._cancel_event = event
            started = time.perf_counter()
            first_chunk = None
            sources = []
            # 手动指定 mode 优先；带图片时直接走视觉；否则交给模态路由
            selected_mode = mode or ("vision" if frames is not None else self.router.route(query))
            if selected_mode not in {"text", "vision"}:
                raise ValueError("mode 必须是 text 或 vision")
            clean_query = self._clean_query(query)
            answer = ""
            observation = ""
            generation_model = self.text_llm
            fused_vision = False

            def emit(chunk):
                """内部回调：记录首段延迟并转交界面（取消后丢弃）。"""
                nonlocal first_chunk
                if event.is_set() or not chunk:
                    return
                if first_chunk is None:
                    first_chunk = time.perf_counter() - started
                if on_chunk:
                    on_chunk(chunk)

            def generate(prompt, context):
                chunks = []
                if not event.is_set():
                    stream = self.text_llm.stream_generate(
                        prompt, kb_context=context, kb_mode=bool(sources),
                        history=list(self.history),
                    )
                    try:
                        for chunk in stream:
                            if event.is_set():
                                break
                            chunks.append(chunk)
                            emit(chunk)
                            if event.is_set():
                                break
                    finally:
                        close = getattr(stream, "close", None)
                        if close:
                            close()
                return "".join(chunks).strip()

            if not event.is_set():
                LOGGER.info("会话记忆使用：历史%d条，显式事实%d条，指代承接=%s",
                            len(self.history), len(self._memory.facts),
                            bool(self._memory.topic and self._memory.is_followup(clean_query)))

            if not event.is_set() and selected_mode == "vision":
                # 视觉路径：两帧输入 + 既有历史（支持多轮视觉追问）
                selected_frames = frames if frames is not None else self.video_capture.recent_frames(2)
                observation = self.vl_model.analyze_frames(
                    selected_frames, self._memory.prompt(clean_query or "描述画面内容") +
                    "\n请先说明画面主体、可辨认的名称或文字和外观，再回应问题；无法确认的识别须保留不确定性。",
                    history=list(self.history),
                )
                if event.is_set():
                    answer = ""
                else:
                    retrieval_query = f"{clean_query}\n{observation[:1600]}"
                    LOGGER.info("视觉检索：结合用户问题与视觉观察检索知识库")
                    sources = self.rag.retrieve(retrieval_query, k=3)
                    from .rag import KB_HIT_DISTANCE
                    sources = [item for item in sources
                               if ('distance' not in item and 'keyword_score' not in item)
                               or item.get('keyword_score', 0) > 0
                               or float(item.get('distance', float('inf'))) <= KB_HIT_DISTANCE]
                    if sources and not event.is_set():
                        fused_vision = True
                        LOGGER.info("视觉融合：已检索到%d条资料，开始生成最终回答", len(sources))
                        prompt = (
                            f"用户问题：{clean_query or '描述画面内容'}\n\n"
                            f"【视觉观察】\n{observation}\n\n"
                            "结合视觉观察与知识库资料回答用户问题。明确区分画面观察、知识库资料及其来源，"
                            "引用资料编号；视觉识别中不确定的部分仍须明确不确定，不得因知识库命中而视为已证实。"
                            "资料与观察不一致时说明差异，不要编造画面中未观察到的信息。"
                            "用途、参数和操作注意事项仅使用参考资料明确给出的信息，不自行扩展其他用途或精度。"
                        )
                        answer = generate(self._memory.prompt(prompt, include_topic=False), self.rag.build_context(sources))
                    elif not event.is_set():
                        generation_model = self.vl_model
                        answer = observation
                        LOGGER.info("视觉检索：未命中资料，使用视觉观察回答")
                        emit(answer)
            elif not event.is_set():
                # 文本路径：先检索（决定是否带知识库上下文），再流式生成
                sources = self.rag.retrieve(self._memory.retrieval_query(clean_query), k=3)
                context = self.rag.build_context(sources)
                answer = generate(self._memory.prompt(clean_query), context)
            answer = to_plain_text(answer).strip()
            # 只有完整完成（未取消且有内容）的回答才进入历史，且只保留最近 12 条
            if answer and not event.is_set():
                self.history.extend([
                    {"role": "user", "content": clean_query},
                    {"role": "assistant", "content": answer},
                ])
                self.history = self.history[-12:]
                self._memory.remember(clean_query, observation=observation, sources=sources)
            if on_details:
                # 汇总推理指标：取客户端最后一条 timings/usage 换算 token/TPOT
                client = getattr(generation_model, "client", None)
                from .inference_metrics import response_measurements
                inference = response_measurements(getattr(client, "last_timings", {}), getattr(client, "last_usage", {}),
                                                  time.perf_counter() - started, first_chunk) if client and answer and not event.is_set() else {}
                if selected_mode == "vision" and not fused_vision and inference:
                    # 视觉适配器整段返回、没有 token 级 SSE：TTFT/TPOT 不适用
                    # The vision adapter returns a complete answer rather than token SSE.
                    inference["ttft_seconds"] = inference["tpot_ms"] = None
                on_details(ResponseDetails(
                    mode=selected_mode,
                    elapsed_seconds=time.perf_counter() - started,
                    first_chunk_seconds=first_chunk,
                    cancelled=event.is_set(),
                    sources=[item.copy() for item in sources],
                    inference=inference,
                ))
            return answer
