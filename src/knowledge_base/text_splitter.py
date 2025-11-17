#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分割模块
支持按章节分割和滑动窗口分割
"""

from typing import List, Dict
from dataclasses import dataclass
from .document_loader import Document


@dataclass
class TextChunk:
    """文本块数据类"""
    text: str  # 文本内容
    metadata: Dict[str, any]  # 元数据（章节号、块索引等）
    
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

