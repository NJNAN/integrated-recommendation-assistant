"""知识库：文档解析（PDF/DOCX/TXT/MD）+ 分块 + 向量索引（MiniLM + FAISS）。

DocumentParser 把文件切成 500 字、重叠 50 字的文本块（带来源元数据）；
VectorStore 用 Sentence Transformers 编码为 384 维归一化向量，
FAISS L2 索引做相似检索，支持保存/恢复（Windows 中文路径安全版）。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .knowledge_search import KeywordIndex


class DocumentParser:
    """多格式文档解析器：统一输出「带元数据的文本块」列表。"""

    CHUNK_SIZE = 500  # 每块目标字数
    CHUNK_OVERLAP = 50  # 相邻块重叠字数，避免语义被截断

    def parse(self, file_path: str | Path) -> list[dict]:
        """按扩展名分发到具体的解析器；不支持的格式直接报错。"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(path)
        if suffix == ".docx":
            return self.parse_docx(path)
        if suffix in {".txt", ".md"}:
            return self.parse_txt(path)
        raise ValueError(f"不支持的文档格式: {suffix}；仅支持 PDF/TXT/MD/DOCX")

    def parse_pdf(self, file_path: str | Path) -> list[dict]:
        """解析 PDF：逐页提取文字，页号带进元数据（便于回答标注来源页）。"""
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise RuntimeError("缺少 PyMuPDF") from exc
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        chunks: list[dict] = []
        with fitz.open(path) as document:
            for page_index in range(document.page_count):
                text = document.load_page(page_index).get_text("text")
                chunks.extend(
                    self.chunk_text(
                        text,
                        {"source": path.name, "path": str(path), "page": page_index + 1},
                    )
                )
        return chunks

    def parse_txt(self, file_path: str | Path) -> list[dict]:
        """解析纯文本：依次尝试常见中文编码，保证读得进来。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        text = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")  # 兜底：坏字节用替代符
        return self.chunk_text(
            text, {"source": path.name, "path": str(path), "page": None}
        )

    def parse_docx(self, file_path: str | Path) -> list[dict]:
        """解析 DOCX：段落 + 表格（表格按行拼成制表符分隔文本）。"""
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("缺少 python-docx") from exc
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        document = Document(path)
        parts = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        for table in document.tables:
            for row in table.rows:
                line = "\t".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if line:
                    parts.append(line)
        return self.chunk_text(
            "\n".join(parts),
            {"source": path.name, "path": str(path), "page": None},
        )

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """把长文本切成带重叠的块。

        切分点优先选「半块位置之后最近的换行/句号」处，实在没有就硬切；
        下一块的起点回退 CHUNK_OVERLAP 字，保证跨块语义连续。
        """
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not cleaned:
            return []
        chunks = []
        start = 0
        index = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + self.CHUNK_SIZE)
            if end < len(cleaned):
                # 在半块位置之后寻找自然边界（换行或句号）
                boundary = max(
                    cleaned.rfind("\n", start + self.CHUNK_SIZE // 2, end),
                    cleaned.rfind("。", start + self.CHUNK_SIZE // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            content = cleaned[start:end].strip()
            if content:
                chunks.append({**metadata, "chunk_index": index, "text": content})
                index += 1
            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.CHUNK_OVERLAP)
        return chunks


class VectorStore:
    """384 维归一化嵌入 + FAISS L2 索引的本地向量库（线程安全）。"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_model=None,
        faiss_module=None,
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self._faiss = faiss_module
        self.index = None
        self.documents: list[dict] = []
        self.dimension = 384
        self._keywords = None
        self._lock = threading.RLock()  # 导入/检索/保存互斥

    def _ensure_backend(self) -> None:
        """惰性加载嵌入模型与 FAISS，并校验维度。"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("缺少 sentence-transformers") from exc
            self.embedding_model = SentenceTransformer(self.model_name)
        if self._faiss is None:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("缺少 faiss-cpu") from exc
            self._faiss = faiss
        # 兼容不同版本的维度接口名
        get_dimension = getattr(self.embedding_model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self.embedding_model.get_sentence_embedding_dimension
        dimension = int(get_dimension())
        if dimension != 384:
            raise ValueError(f"嵌入模型维度应为 384，实际为 {dimension}")
        self.dimension = dimension
        if self.index is None:
            self.index = self._faiss.IndexFlatL2(self.dimension)  # 精确 L2 检索

    def add_documents(self, documents: list[dict]) -> int:
        """追加文档块：编码为归一化向量加入索引。返回新增块数。"""
        valid = [doc.copy() for doc in documents if doc.get("text", "").strip()]
        if not valid:
            return 0
        with self._lock:
            self._ensure_backend()
            embeddings = self.embedding_model.encode(
                [doc["text"] for doc in valid],
                convert_to_numpy=True,
                normalize_embeddings=True,  # 归一化后 L2 距离与余弦一致
                show_progress_bar=False,
            ).astype("float32")
            self.index.add(embeddings)
            self.documents.extend(valid)
            self._keywords = None
            return len(valid)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """综合中文关键词、文件名与语义向量排序，保留真实 L2 距离。"""
        if not query.strip() or k <= 0:
            return []
        with self._lock:
            if not self.documents:
                return []
            self._ensure_backend()
            vector = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            # Flat 索引本来就扫描全库；保留所有距离，避免关键词候选被 top-k 截掉。
            distances, indices = self.index.search(vector, len(self.documents))
            if self._keywords is None:
                self._keywords = KeywordIndex(self.documents)
            lexical = self._keywords.scores(query)
            maximum = max(lexical, default=0)
            results = []
            for distance, index in zip(distances[0], indices[0]):
                if index < 0:
                    continue  # FAISS 未命中时返回 -1
                keyword_score = lexical[int(index)]
                semantic = max(0.0, min(1.0, 1 - float(distance) / 2))
                score = .35 * semantic + .65 * keyword_score / maximum if maximum else semantic
                results.append({**self.documents[int(index)], "distance": float(distance),
                                "keyword_score": keyword_score, "retrieval_score": score})
            results.sort(key=lambda r: r['retrieval_score'], reverse=True)
            return results[:k]

    def source_catalog(self, word_only=False):
        """返回已经导入的实际资料，按路径去重。"""
        with self._lock:
            sources = {}
            for doc in self.documents:
                source = doc.get('source', '未知来源')
                if word_only and Path(source).suffix.lower() not in {'.doc', '.docx'}:
                    continue
                key = doc.get('path') or source
                if key not in sources:
                    sources[key] = {'source': source, 'path': doc.get('path'), 'chunks': 0}
                sources[key]['chunks'] += 1
            return sorted(sources.values(), key=lambda s: (s['source'], s['path'] or ''))

    def clear(self) -> None:
        """清空索引与文档（重新建库时用）。"""
        with self._lock:
            self.documents.clear()
            self.index = None
            self._keywords = None

    def save(self, directory: str | Path) -> None:
        """把索引与文档元数据保存到目录。

        注意：FAISS 的 Windows 文件接口打不开含中文字符的路径，
        所以这里「先在内存里序列化」再用 pathlib 写文件。
        """
        with self._lock:
            self._ensure_backend()
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            # FAISS' Windows file API cannot open paths containing CJK characters.
            # Serializing in memory lets pathlib handle the actual filesystem I/O.
            serialized = self._faiss.serialize_index(self.index)
            (directory / "index.faiss").write_bytes(serialized.tobytes())
            (directory / "documents.json").write_text(
                json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def load(self, directory: str | Path) -> int:
        """从目录恢复索引与文档；校验数量/维度一致。返回恢复的块数（0=无存档）。"""
        with self._lock:
            directory = Path(directory)
            index_path = directory / "index.faiss"
            docs_path = directory / "documents.json"
            if not index_path.exists() or not docs_path.exists():
                return 0
            self._ensure_backend()
            import numpy as np

            serialized = np.frombuffer(index_path.read_bytes(), dtype="uint8")
            index = self._faiss.deserialize_index(serialized)
            documents = json.loads(docs_path.read_text(encoding="utf-8"))
            # 防御性校验：坏存档宁可报错，也不要带着不一致的索引去检索
            if not isinstance(documents, list) or any(
                not isinstance(doc, dict) or not isinstance(doc.get("text"), str)
                for doc in documents
            ):
                raise ValueError("知识库文档元数据格式无效")
            if index.ntotal != len(documents):
                raise ValueError("FAISS 索引与文档元数据数量不一致")
            if index.d != self.dimension:
                raise ValueError("FAISS 索引与嵌入模型维度不一致")
            self.index = index
            self.documents = documents
            self._keywords = None
            return len(self.documents)
