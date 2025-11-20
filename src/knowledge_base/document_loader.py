#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档加载模块
支持加载TXT格式的小说章节文件，并提取章节信息
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """文档数据类"""
    content: str  # 文档内容
    metadata: Dict[str, any]  # 元数据（文件名、章节号等）
    
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

