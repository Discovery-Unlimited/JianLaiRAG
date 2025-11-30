#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库构建脚本 - Kaggle 版本
整合所有模块，实现完整的知识库构建流程
适配 Kaggle 环境，硬编码路径和配置
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Any, Dict, Union
import sys
import pickle
import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ==================== Kaggle 环境配置（硬编码） ====================
# 输入路径
KAGGLE_INPUT_PATH = "/kaggle/input/jianlainovel/raw/"

# 输出路径
KAGGLE_VECTOR_DB_PATH = "/kaggle/working/storage/vector_db"
KAGGLE_CACHE_DIR = "/kaggle/working/storage/cache"
KAGGLE_LOG_FILE = "/kaggle/working/build_kb.log"

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # 直接从 HuggingFace 下载
EMBEDDING_DEVICE = "cuda"  # 可选: "cuda" 或 "cpu"
EMBEDDING_BATCH_SIZE = 32

# 向量数据库配置
VECTOR_DB_COLLECTION_NAME = "jianlai_novel"

# 文本分割配置
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 50
TEXT_SPLIT_BY_CHAPTER = True
TEXT_MIN_CHUNK_SIZE = 100

# 日志配置
LOG_LEVEL = "INFO"
# ==================================================================


# ==================== 文档加载模块 ====================
@dataclass
class Document:
    """文档数据类"""
    content: str  # 文档内容
    metadata: Dict[str, Any]  # 元数据（文件名、章节号等）
    
    def __repr__(self):
        return f"Document(chapter={self.metadata.get('chapter', 'unknown')}, length={len(self.content)})"


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, encoding: str = "utf-8", logger: Optional[logging.Logger] = None):
        """
        初始化文档加载器
        
        Args:
            encoding: 文件编码，默认为utf-8
            logger: 日志记录器，如果为None则创建默认logger
        """
        self.encoding = encoding
        # 匹配章节号的正则表达式，支持"第X章"、"第XXX章"等格式
        self.chapter_pattern = re.compile(r'第(\d+)章')
        self.logger = logger or logging.getLogger(__name__)
    
    def load_file(self, file_path: str) -> Optional[Document]:
        """
        加载单个TXT文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document对象，如果加载失败返回None
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read().strip()
            
            if not content:
                return None
            
            # 从文件名提取章节号
            filename = file_path.stem  # 不含扩展名的文件名
            chapter_match = self.chapter_pattern.search(filename)
            chapter_num = None
            if chapter_match:
                chapter_num = int(chapter_match.group(1))
            
            # 提取章节标题（文件名中下划线后的部分）
            title = None
            if '_' in filename:
                title = filename.split('_', 1)[1]
            
            # 构建元数据
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'chapter': chapter_num,
                'title': title,
                'source': 'jianlai_novel'
            }
            
            return Document(content=content, metadata=metadata)
            
        except Exception as e:
            self.logger.warning(f"加载文件失败 {file_path}: {e}")
            return None
    
    def load_directory(self, directory_path: str, pattern: str = "*.txt") -> List[Document]:
        """
        批量加载目录中的所有TXT文件
        
        Args:
            directory_path: 目录路径
            pattern: 文件匹配模式，默认为"*.txt"
            
        Returns:
            Document对象列表，按章节号排序
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        documents = []
        txt_files = sorted(directory.glob(pattern))
        
        self.logger.info(f"找到 {len(txt_files)} 个TXT文件")
        
        for file_path in txt_files:
            doc = self.load_file(file_path)
            if doc:
                documents.append(doc)
        
        # 按章节号排序
        documents.sort(key=lambda x: x.metadata.get('chapter', 0) or 0)
        
        self.logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def extract_chapter_number(self, filename: str) -> Optional[int]:
        """
        从文件名提取章节号
        
        Args:
            filename: 文件名
            
        Returns:
            章节号，如果未找到返回None
        """
        match = self.chapter_pattern.search(filename)
        if match:
            return int(match.group(1))
        return None
# =====================================================


# ==================== 文本分割模块 ====================
@dataclass
class TextChunk:
    """文本块数据类"""
    text: str  # 文本内容
    metadata: Dict[str, Any]  # 元数据（章节号、块索引等）
    
    def __repr__(self):
        return f"TextChunk(chapter={self.metadata.get('chapter', 'unknown')}, chunk_idx={self.metadata.get('chunk_idx', 0)}, length={len(self.text)})"


class TextSplitter:
    """文本分割器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        split_by_chapter: bool = True,
        min_chunk_size: int = 100
    ):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
            split_by_chapter: 是否按章节分割（优先保持章节边界）
            min_chunk_size: 最小文本块大小（小于此值不分割）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by_chapter = split_by_chapter
        self.min_chunk_size = min_chunk_size
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
    
    def split_document(self, document: Document) -> List[TextChunk]:
        """
        分割单个文档
        
        Args:
            document: Document对象
            
        Returns:
            TextChunk对象列表
        """
        text = document.content
        metadata = document.metadata.copy()
        
        # 如果按章节分割且文本长度小于chunk_size，直接返回
        if self.split_by_chapter and len(text) <= self.chunk_size:
            return [TextChunk(
                text=text,
                metadata={**metadata, 'chunk_idx': 0, 'total_chunks': 1}
            )]
        
        # 对于超长章节，使用滑动窗口分割
        if len(text) > self.chunk_size:
            chunks = self._sliding_window_split(text, metadata)
        else:
            # 文本较短，不需要分割
            chunks = [TextChunk(
                text=text,
                metadata={**metadata, 'chunk_idx': 0, 'total_chunks': 1}
            )]
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        批量分割文档列表
        
        Args:
            documents: Document对象列表
            
        Returns:
            TextChunk对象列表
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _sliding_window_split(self, text: str, base_metadata: Dict) -> List[TextChunk]:
        """
        使用滑动窗口分割文本
        
        Args:
            text: 要分割的文本
            base_metadata: 基础元数据
            
        Returns:
            TextChunk对象列表
        """
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = start + self.chunk_size
            
            # 如果剩余文本不足一个chunk_size，直接取到末尾
            if end >= len(text):
                chunk_text = text[start:]
            else:
                # 尝试在句号、问号、感叹号处断开，避免截断句子
                chunk_text = text[start:end]
                # 向后查找最近的句号、问号、感叹号或换行符
                for punct in ['。', '！', '？', '\n', '.', '!', '?']:
                    last_punct = chunk_text.rfind(punct)
                    if last_punct > self.chunk_size * 0.7:  # 至少保留70%的chunk_size
                        chunk_text = text[start:start + last_punct + 1]
                        end = start + last_punct + 1
                        break
            
            # 过滤掉太小的块（除非是最后一个块）
            if len(chunk_text.strip()) >= self.min_chunk_size or end >= len(text):
                metadata = base_metadata.copy()
                metadata['chunk_idx'] = chunk_idx
                chunks.append(TextChunk(text=chunk_text.strip(), metadata=metadata))
                chunk_idx += 1
            
            # 计算下一个块的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        # 更新每个块的total_chunks信息
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata['total_chunks'] = total_chunks
        
        return chunks
    
    def split_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        直接分割文本字符串（不通过Document对象）
        
        Args:
            text: 要分割的文本
            metadata: 可选的元数据
            
        Returns:
            TextChunk对象列表
        """
        if metadata is None:
            metadata = {}
        
        if len(text) <= self.chunk_size:
            return [TextChunk(
                text=text,
                metadata={**metadata, 'chunk_idx': 0, 'total_chunks': 1}
            )]
        
        return self._sliding_window_split(text, metadata)
# =====================================================


# ==================== 嵌入模型模块 ====================
class Embedder:
    """嵌入模型封装类"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        local_model_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: HuggingFace模型名称或本地路径
            local_model_dir: 本地优先加载的目录（若存在则直接使用）
            device: 设备类型，"cuda"或"cpu"，如果为None则自动选择
            batch_size: 批量处理的大小
            logger: 日志记录器，如果为None则创建默认logger
        """
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.batch_size = batch_size
        self.model = None
        self._model_source = None
        self._embedding_dim = None
        self.logger = logger or logging.getLogger(__name__)
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        self._load_model()
    
    def _resolve_model_source(self) -> str:
        """
        决定优先加载的模型路径：当前配置 -> 本地目录 -> HuggingFace 名称
        """
        configured_path = Path(self.model_name)
        if configured_path.exists():
            return str(configured_path.resolve())
        
        if self.local_model_dir:
            local_dir = Path(self.local_model_dir)
            if local_dir.exists():
                return str(local_dir.resolve())
        
        return self.model_name
    
    def _load_model(self):
        """加载嵌入模型（本地优先，缺失则自动下载）"""
        try:
            resolved_model = self._resolve_model_source()
            self._model_source = resolved_model
            self.logger.info(f"正在加载嵌入模型: {resolved_model}")
            
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("⚠️  警告：要求使用CUDA，但当前环境不可用，自动切换为CPU")
                self.device = "cpu"
            
            self.model = SentenceTransformer(resolved_model, device=self.device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"模型加载完成（来源: {resolved_model}），设备: {self.device}")
            
            if self.device == "cuda":
                try:
                    test_tensor = torch.randn(1).to(self.device)
                    self.logger.info(f"✅ GPU验证成功: {test_tensor.device}")
                except Exception as e:
                    self.logger.warning(f"⚠️  GPU验证失败: {e}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 单个文本字符串或文本列表
            
        Returns:
            numpy数组，形状为 (n, dim) 或 (dim,)，其中n是文本数量，dim是向量维度
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        # 使用sentence-transformers批量编码
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
        
        # 合并所有批次
        result = np.vstack(embeddings)
        
        # 如果输入是单个文本，返回一维数组
        if single_text:
            return result[0]
        
        return result
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        批量向量化文本列表
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            numpy数组，形状为 (n, dim)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    @property
    def dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            向量维度
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            return self._embedding_dim
        test_embedding = self.embed("test")
        self._embedding_dim = len(test_embedding)
        return self._embedding_dim
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        使Embedder对象可调用
        
        Args:
            texts: 单个文本字符串或文本列表
            
        Returns:
            numpy数组
        """
        return self.embed(texts)
# =====================================================


# ==================== 向量数据库模块 ====================
class VectorStore:
    """向量数据库操作类"""
    
    def __init__(
        self,
        persist_directory: str = "storage/vector_db",
        collection_name: str = "jianlai_novel"
    ):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 持久化目录路径
            collection_name: 集合名称
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # 创建持久化目录
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = None
        self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"加载现有集合: {self.collection_name}")
        except Exception:
            # 集合不存在，创建新集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "剑来小说知识库"}
            )
            print(f"创建新集合: {self.collection_name}")
    
    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ):
        """
        添加文本块和向量到数据库
        
        Args:
            chunks: TextChunk对象列表
            embeddings: 对应的向量数组，形状为 (n, dim)
            batch_size: 批量添加的大小
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks数量({len(chunks)})与embeddings数量({len(embeddings)})不匹配")
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        vectors = []
        
        for i, chunk in enumerate(chunks):
            # 生成唯一ID：优先使用文件名 + 块索引，避免章节号重复导致冲突
            chapter = chunk.metadata.get('chapter', 'unknown')
            chunk_idx = chunk.metadata.get('chunk_idx', i)
            filename = chunk.metadata.get('filename')

            if filename:
                # 使用文件名（去掉扩展名）+ 块索引
                name_stem = Path(filename).stem.replace(' ', '_')
                chunk_id = f"{name_stem}_idx{chunk_idx}"
            else:
                chunk_id = f"ch{chapter}_idx{chunk_idx}_{i}"

            ids.append(chunk_id)
            
            # 文档内容
            documents.append(chunk.text)
            
            # 元数据（Chroma要求元数据值必须是字符串、数字或布尔值）
            metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)
            
            # 向量（转换为列表）
            vectors.append(embeddings[i].tolist())
        
        # 批量添加
        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_vectors
            )
            
            print(f"已添加 {min(i + batch_size, total)}/{total} 个文本块")
        
        print(f"成功添加 {total} 个文本块到向量数据库")
    
    def get_collection_info(self) -> Dict:
        """
        获取集合信息
        
        Returns:
            包含集合信息的字典
        """
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'count': count,
            'path': str(self.persist_directory)
        }
    
    def delete_collection(self):
        """删除集合（谨慎使用）"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"已删除集合: {self.collection_name}")
            # 重新创建空集合
            self._get_or_create_collection()
        except Exception as e:
            print(f"删除集合失败: {e}")
    
    def reset(self):
        """重置集合（清空所有数据）"""
        self.delete_collection()
        self._get_or_create_collection()
        print("集合已重置")
# =====================================================


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, force: bool = False) -> logging.Logger:
    """
    设置日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        force: 是否强制重新配置（用于重新设置日志时）
    """
    # 如果强制重新配置，先清除现有的处理器
    if force:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # 检查是否已经配置过
    if not force and logging.getLogger().handlers:
        # 已经配置过，只添加文件处理器（如果指定了）
        if log_file:
            root_logger = logging.getLogger()
            # 检查是否已经有文件处理器
            has_file_handler = any(
                isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                for h in root_logger.handlers
            )
            if not has_file_handler:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(getattr(logging, log_level.upper()))
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                root_logger.addHandler(file_handler)
                root_logger.setLevel(getattr(logging, log_level.upper()))
    else:
        # 首次配置
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        
        # 使用 basicConfig 配置（仅在未配置时生效）
        # 如果 force=True，已经清除了处理器，所以这里会重新配置
        try:
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=handlers
            )
        except ValueError:
            # 如果已经配置过且 force=False，basicConfig 会抛出 ValueError
            # 这种情况下手动添加处理器
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_level.upper()))
            for handler in handlers:
                root_logger.addHandler(handler)
    
    return logging.getLogger(__name__)


def build_chunks_signature(chunks_list: List[Any]) -> str:
    """
    构建文本块的签名，用于缓存验证
    
    Args:
        chunks_list: 文本块列表
        
    Returns:
        MD5哈希签名
    """
    hasher = hashlib.md5()
    for chunk in chunks_list:
        part = (
            f"{chunk.metadata.get('filename', '')}-"
            f"{chunk.metadata.get('chapter', '')}-"
            f"{chunk.metadata.get('chunk_idx', 0)}-"
            f"{len(chunk.text)}"
        )
        hasher.update(part.encode('utf-8', errors='ignore'))
    return hasher.hexdigest()


def load_embeddings_cache(
    cache_file: Path,
    chunks_signature: str,
    reset: bool,
    logger: logging.Logger
) -> Optional[List[Any]]:
    """
    加载缓存的嵌入向量
    
    Args:
        cache_file: 缓存文件路径
        chunks_signature: 文本块签名
        reset: 是否重置缓存
        logger: 日志记录器
        
    Returns:
        缓存的嵌入向量，如果不存在或签名不匹配则返回None
    """
    if not cache_file.exists() or reset:
        return None
    
    try:
        with cache_file.open('rb') as f:
            cache_data = pickle.load(f)
        if cache_data.get('signature') == chunks_signature:
            logger.info(f"检测到缓存嵌入文件，跳过重新向量化: {cache_file}")
            return cache_data.get('embeddings')
        else:
            logger.info("缓存签名与当前文本不匹配，将重新计算嵌入")
            return None
    except Exception as e:
        logger.warning(f"加载嵌入缓存失败，将重新计算: {e}")
        return None


def save_embeddings_cache(
    cache_file: Path,
    chunks_signature: str,
    embeddings: List[Any],
    logger: logging.Logger
) -> bool:
    """
    保存嵌入向量到缓存
    
    Args:
        cache_file: 缓存文件路径
        chunks_signature: 文本块签名
        embeddings: 嵌入向量列表
        logger: 日志记录器
        
    Returns:
        是否成功保存
    """
    try:
        with cache_file.open('wb') as f:
            pickle.dump({
                'signature': chunks_signature,
                'embeddings': embeddings
            }, f)
        logger.info(f"嵌入结果已缓存到: {cache_file}")
        return True
    except Exception as e:
        logger.warning(f"写入嵌入缓存失败: {e}")
        return False


def build_knowledge_base(
    data_path: Optional[str] = None,
    reset: bool = False
) -> bool:
    """
    构建知识库
    
    Args:
        data_path: 数据源路径（如果为None，使用默认Kaggle输入路径）
        reset: 是否重置向量数据库
        
    Returns:
        是否成功构建
    """
    # 初始化日志
    logger = setup_logging(LOG_LEVEL, KAGGLE_LOG_FILE)
    
    # 数据源路径
    if data_path is None:
        data_path = KAGGLE_INPUT_PATH
    
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.error(f"数据路径不存在: {data_path}")
        return False
    
    logger.info("=" * 60)
    logger.info("开始构建知识库（Kaggle 环境）")
    logger.info("=" * 60)
    logger.info(f"输入路径: {data_path}")
    logger.info(f"向量数据库路径: {KAGGLE_VECTOR_DB_PATH}")
    logger.info(f"缓存目录: {KAGGLE_CACHE_DIR}")
    logger.info(f"日志文件: {KAGGLE_LOG_FILE}")
    logger.info(f"嵌入模型: {EMBEDDING_MODEL_NAME} (从 HuggingFace 下载)")
    
    # 1. 加载文档
    logger.info("\n[步骤 1/5] 加载文档...")
    try:
        loader = DocumentLoader(logger=logger)
        documents = loader.load_directory(data_path)
        if not documents:
            logger.error("未找到任何文档，请检查数据路径")
            return False
        # 成功加载的日志已在 DocumentLoader 中记录
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        return False
    
    # 2. 分割文本
    logger.info("\n[步骤 2/5] 分割文本...")
    try:
        splitter = TextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP,
            split_by_chapter=TEXT_SPLIT_BY_CHAPTER,
            min_chunk_size=TEXT_MIN_CHUNK_SIZE
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            logger.error("文本分割后未生成任何文本块")
            return False
        logger.info(f"文本分割完成，共生成 {len(chunks)} 个文本块")
    except Exception as e:
        logger.error(f"文本分割失败: {e}")
        return False

    # 缓存目录与文件（用于在步骤5失败后复用嵌入结果）
    cache_dir = Path(KAGGLE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{VECTOR_DB_COLLECTION_NAME}_embeddings.pkl"

    chunks_signature = build_chunks_signature(chunks)
    
    # 3. 初始化嵌入模型
    logger.info("\n[步骤 3/5] 初始化嵌入模型...")
    try:
        # 直接使用 HuggingFace 模型名，不使用本地目录
        embedder = Embedder(
            model_name=EMBEDDING_MODEL_NAME,
            local_model_dir=None,  # Kaggle 环境直接从 HuggingFace 下载
            device=EMBEDDING_DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
            logger=logger
        )
        logger.info(f"嵌入模型初始化完成，向量维度: {embedder.dimension}")
    except Exception as e:
        logger.error(f"初始化嵌入模型失败: {e}")
        return False
    
    # 4. 向量化文本块（支持缓存）
    logger.info("\n[步骤 4/5] 向量化文本块...")
    cached_embeddings = load_embeddings_cache(cache_file, chunks_signature, reset, logger)
    
    if cached_embeddings is not None:
        embeddings = cached_embeddings
        logger.info("使用缓存嵌入完成步骤 4/5")
    else:
        try:
            texts = [chunk.text for chunk in chunks]
            logger.info(f"开始向量化 {len(texts)} 个文本块...")
            embeddings = embedder.embed_batch(texts, show_progress=True)
            logger.info(f"向量化完成，生成 {len(embeddings)} 个向量")
            
            # 保存缓存
            save_embeddings_cache(cache_file, chunks_signature, embeddings, logger)
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            return False
    
    # 验证数据一致性
    if len(chunks) != len(embeddings):
        logger.error(
            f"数据不一致：文本块数量 ({len(chunks)}) 与嵌入向量数量 ({len(embeddings)}) 不匹配"
        )
        return False
    
    # 5. 存储到向量数据库
    logger.info("\n[步骤 5/5] 存储到向量数据库...")
    try:
        vector_store = VectorStore(
            persist_directory=KAGGLE_VECTOR_DB_PATH,
            collection_name=VECTOR_DB_COLLECTION_NAME
        )

        logger.warning("清理旧向量集合，准备重建...")
        vector_store.reset()
        
        vector_store.add_chunks(chunks, embeddings)
        
        # 构建成功后清理缓存，避免下次误用旧数据
        try:
            if cache_file.exists():
                cache_file.unlink()
                logger.info("已清理临时嵌入缓存文件")
        except Exception as e:
            logger.warning(f"清理嵌入缓存失败，请手动删除 {cache_file}: {e}")
        
        # 显示统计信息
        info = vector_store.get_collection_info()
        logger.info("\n" + "=" * 60)
        logger.info("知识库构建完成！")
        logger.info("=" * 60)
        logger.info(f"集合名称: {info['name']}")
        logger.info(f"文本块数量: {info['count']}")
        logger.info(f"存储路径: {info['path']}")
        logger.info("=" * 60)
        
        return True
    except Exception as e:
        logger.error(f"存储到向量数据库失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='构建RAG知识库（Kaggle版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_kb_kaggle.py
  # 使用自定义数据路径和重置数据库
  python build_kb_kaggle.py --data /kaggle/input/jianlainovel/raw/ --reset
  
注意: 在 Jupyter Notebook 中，建议直接导入并调用函数:
  from build_kb_kaggle import build_knowledge_base
  build_knowledge_base()
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help=f'数据源路径（默认: {KAGGLE_INPUT_PATH}）'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='重置向量数据库（清空现有数据）'
    )
    
    # 使用 parse_known_args 来忽略未知参数（如 Jupyter 传递的 -f 参数）
    # 这样可以兼容 Jupyter/Colab 环境，它们会自动传递 -f 参数给脚本
    args, unknown = parser.parse_known_args()
    
    # 静默忽略未知参数（Jupyter 会传递 -f 参数，这是正常的）
    
    try:
        success = build_knowledge_base(
            data_path=args.data,
            reset=args.reset
        )
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n构建知识库时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

