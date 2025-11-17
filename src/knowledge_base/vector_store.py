#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库操作模块
使用Chroma实现向量数据库的创建、保存和加载
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from .text_splitter import TextChunk


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
            # 生成唯一ID：章节号_块索引
            chapter = chunk.metadata.get('chapter', 'unknown')
            chunk_idx = chunk.metadata.get('chunk_idx', i)
            chunk_id = f"ch{chapter}_idx{chunk_idx}"
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
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索相似文本块
        
        Args:
            query_embedding: 查询向量，形状为 (dim,) 或 (1, dim)
            n_results: 返回结果数量
            where: 可选的过滤条件（元数据过滤）
            
        Returns:
            结果列表，每个结果包含文档、元数据和距离
        """
        # 确保query_embedding是一维数组
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]
        
        query_vector = query_embedding.tolist()
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where
        )
        
        # 格式化结果
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
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

