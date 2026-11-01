"""RAG 管线：文档导入 → 向量检索 → 拼接参考资料 → 增强生成。

RAGPipeline 把知识库（VectorStore）与文本模型（TextLLM）串起来：
import_document() 解析并入库（可自动存盘）；answer() / stream_answer() 先检索
再让模型「依据资料」回答（资料不足时如实声明）。
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from .knowledge_base import DocumentParser, VectorStore

LOGGER = logging.getLogger(__name__)

# 命中判定阈值（归一化嵌入下的 L2 距离；越小越相关）。
# 实测本机知识库：库内问题 top1 距离 0.46~0.62，库外问题 0.94~1.01，
# 因此取 0.80 作为「命中 / 未命中」的日志判定线。它只用于日志与观察，
# 不过滤检索结果——是否足以回答仍由提示词约束模型如实说明。
KB_HIT_DISTANCE = float(os.getenv("MMAI_KB_HIT_DISTANCE", "0.80"))


class RAGPipeline:
    """检索增强生成：把检索到的 top-k 资料块注入提示词。"""

    def __init__(
        self,
        vector_store: VectorStore,
        text_llm,
        parser: DocumentParser | None = None,
        index_dir: str | Path | None = None,
    ):
        self.vector_store = vector_store
        self.text_llm = text_llm
        self.parser = parser or DocumentParser()
        self.index_dir = Path(index_dir) if index_dir is not None else None

    def import_document(self, file_path: str | Path) -> int:
        """导入一个文档：解析→分块→入索引→（配置了 index_dir 时）保存。返回块数。"""
        documents = self.parser.parse(file_path)
        if not documents:
            raise ValueError("文档没有可提取的文字；扫描版 PDF 请先进行 OCR")
        count = self.vector_store.add_documents(documents)
        if count and self.index_dir is not None:
            self.vector_store.save(self.index_dir)
        LOGGER.info('导入文档完成：成功 1 个（%d 个知识块）| 文件=%s', count, Path(file_path).name)
        return count

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """检索与问题最相关的 k 个资料块，并把「命中 / 未命中」写入日志。"""
        if k <= 0:
            return []
        normalized = re.sub(r'[^\w\u3400-\u9fff]', '', query.casefold())
        inventory = re.fullmatch(
            r'(?:请问)?(?:你的|当前|本地|这个)?知识库(?:里面|里|中)?(?:都)?'
            r'(?:有|包含|收录)(?:哪些|什么|多少)(?:word|文档|文件|资料|知识|内容)*(?:呢|吗)?',
            normalized,
        )
        if inventory:
            sources = self.vector_store.source_catalog(word_only='word' in normalized)
            label = 'Word 文档' if 'word' in normalized else '资料'
            lines = [f'当前已导入 {len(sources)} 份{label}。以下是实际文件目录：']
            lines.extend(f"{i}. {item['source']}（{item['chunks']} 个知识块）"
                         for i, item in enumerate(sources[:100], 1))
            if len(sources) > 100:
                lines.append(f'此处列出前 100 份，另有 {len(sources) - 100} 份。')
            LOGGER.info('知识库目录：已导入 %d 份%s', len(sources), label)
            return [{'source': '知识库资料目录', 'text': '\n'.join(lines),
                     'distance': 0.0, 'retrieval_method': 'catalog'}]
        results = self.vector_store.search(query, k)
        if not results:
            LOGGER.info("知识库检索：未命中（知识库为空或没有可用资料块）| 查询=%s", query)
            return results
        top = results[0]
        distance = float(top.get("distance", float("inf")))
        source = Path(str(top.get("source", "未知来源"))).name
        page = f" 第{top['page']}页" if top.get("page") else ""
        verdict = ('关键词匹配' if top.get('keyword_score', 0) > 0 else
                   '语义命中' if distance <= KB_HIT_DISTANCE else '未命中（语义相关性较低）')
        LOGGER.info(
            "知识库检索：%s | 返回 %d 条 | top1=%.3f（阈值 %.2f）| 来源=%s%s | 查询=%s",
            verdict, len(results), distance, KB_HIT_DISTANCE, source, page, query,
        )
        return results

    @staticmethod
    def build_context(results: list[dict]) -> str:
        """把检索结果拼成「资料 N（来源，第 X 页）」格式的参考上下文。"""
        sections = []
        for index, item in enumerate(results, start=1):
            page = f"，第 {item['page']} 页" if item.get("page") else ""
            sections.append(
                f"资料 {index}（{item.get('source', '未知来源')}{page}）\n{item['text']}"
            )
        return "\n\n".join(sections)

    def answer(self, query: str, k: int = 3) -> str:
        """知识库问答（一次性）：检索 + 增强生成，返回完整回答。"""
        results = self.retrieve(query, k)
        return self.text_llm.generate(
            query,
            kb_context=self.build_context(results),
            kb_mode=True,
        )

    def stream_answer(self, query: str, k: int = 3):
        """知识库问答（流式）：逐段 yield 回答内容。"""
        results = self.retrieve(query, k)
        yield from self.text_llm.stream_generate(
            query,
            kb_context=self.build_context(results),
            kb_mode=True,
        )
