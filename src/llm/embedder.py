#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型封装模块
支持BGE-M3等嵌入模型的向量化
"""

from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """嵌入模型封装类"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        local_model_dir: Optional[str] = "models/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: HuggingFace模型名称或本地路径
            local_model_dir: 本地优先加载的目录（若存在则直接使用）
            device: 设备类型，"cuda"或"cpu"，如果为None则自动选择
            batch_size: 批量处理的大小
        """
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.batch_size = batch_size
        self.model = None
        self._model_source = None
        self._embedding_dim = None
        
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
            print(f"正在加载嵌入模型: {resolved_model}")
            
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  警告：要求使用CUDA，但当前环境不可用，自动切换为CPU")
                self.device = "cpu"
            
            self.model = SentenceTransformer(resolved_model, device=self.device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"模型加载完成（来源: {resolved_model}），设备: {self.device}")
            
            if self.device == "cuda":
                try:
                    test_tensor = torch.randn(1).to(self.device)
                    print(f"✅ GPU验证成功: {test_tensor.device}")
                except Exception as e:
                    print(f"⚠️  GPU验证失败: {e}")
        except Exception as e:
            print(f"加载模型失败: {e}")
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

