#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型封装模块
支持BGE-M3等嵌入模型的向量化
"""

import torch
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os


class Embedder:
    """嵌入模型封装类"""
    
    def __init__(
        self,
        model_name: str = "ollama://bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称或路径
                - "BAAI/bge-m3": 使用sentence-transformers加载BGE-M3
                - "ollama://bge-m3": 使用ollama（需要额外实现）
            device: 设备类型，"cuda"或"cpu"，如果为None则自动选择
            batch_size: 批量处理的大小
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.ollama_model_name = None  # 初始化为None
        self.model = None  # 初始化为None
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            if self.model_name.startswith("ollama://"):
                # 使用ollama API
                self.ollama_model_name = self.model_name.replace("ollama://", "")
                self.ollama_base_url = "http://localhost:11434"
                self.model = None  # ollama不使用sentence-transformers模型对象
                
                # 如果配置了GPU，尝试设置环境变量（虽然对已运行的Ollama服务无效）
                if self.device == "cuda" and torch.cuda.is_available():
                    # 设置环境变量（对当前进程有效，但Ollama服务需要重启才能生效）
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    print(f"⚠️  注意：Ollama服务需要在启动前设置GPU环境变量")
                    print(f"   如果Ollama仍使用CPU，请：")
                    print(f"   1. 停止Ollama服务")
                    print(f"   2. 设置环境变量: set CUDA_VISIBLE_DEVICES=0 (Windows) 或 export CUDA_VISIBLE_DEVICES=0 (Linux)")
                    print(f"   3. 重新启动Ollama服务")
                    print(f"   或运行: python tools/check_ollama_gpu.py 检查GPU使用情况")
                
                # 检查ollama服务是否运行
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        print(f"正在使用Ollama嵌入模型: {self.ollama_model_name}")
                        
                        # 检查Ollama是否使用GPU（通过查看模型信息）
                        try:
                            model_info = requests.post(
                                f"{self.ollama_base_url}/api/show",
                                json={"name": self.ollama_model_name},
                                timeout=5
                            )
                            if model_info.status_code == 200:
                                info = model_info.json()
                                # 检查是否有GPU相关信息
                                details = info.get('details', {})
                                if details:
                                    print(f"   模型格式: {details.get('format', 'unknown')}")
                        except:
                            pass
                        
                        # 测试获取向量维度
                        test_embedding = self._ollama_embed(["test"])
                        self._embedding_dim = len(test_embedding[0])
                        print(f"Ollama模型加载完成，向量维度: {self._embedding_dim}")
                    else:
                        raise ConnectionError(f"Ollama服务响应异常: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    raise ConnectionError(
                        f"无法连接到Ollama服务 ({self.ollama_base_url})。"
                        "请确保Ollama已安装并正在运行。"
                    )
            else:
                # 使用sentence-transformers加载模型
                print(f"正在加载嵌入模型: {self.model_name}")
                
                # 确保使用正确的设备
                if self.device == "cuda" and not torch.cuda.is_available():
                    print(f"⚠️  警告：配置要求使用CUDA，但CUDA不可用，将使用CPU")
                    self.device = "cpu"
                
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.ollama_model_name = None
                print(f"模型加载完成，设备: {self.device}")
                
                # 验证GPU使用
                if self.device == "cuda":
                    try:
                        test_tensor = torch.randn(1).to(self.device)
                        print(f"✅ GPU验证成功: {test_tensor.device}")
                    except Exception as e:
                        print(f"⚠️  GPU验证失败: {e}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        """
        使用Ollama API获取文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.ollama_model_name,
                        "prompt": text
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", [])
                if not embedding:
                    raise ValueError(f"Ollama返回的嵌入向量为空")
                embeddings.append(embedding)
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"调用Ollama API失败: {e}")
        
        return embeddings
    
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
        
        # 使用ollama或sentence-transformers
        if self.ollama_model_name:
            # 使用ollama API
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._ollama_embed(batch)
                embeddings.extend(batch_embeddings)
            
            result = np.array(embeddings, dtype=np.float32)
            # 归一化向量
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-8)
        else:
            # 使用sentence-transformers
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # 归一化向量，便于相似度计算
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
            show_progress: 是否显示进度条（ollama模式下会打印进度）
            
        Returns:
            numpy数组，形状为 (n, dim)
        """
        if self.ollama_model_name:
            # 使用ollama API
            embeddings = []
            total = len(texts)
            for i in range(0, total, self.batch_size):
                batch = texts[i:i + self.batch_size]
                if show_progress:
                    print(f"向量化进度: {min(i + self.batch_size, total)}/{total}", end='\r')
                batch_embeddings = self._ollama_embed(batch)
                embeddings.extend(batch_embeddings)
            
            if show_progress:
                print()  # 换行
            
            result = np.array(embeddings, dtype=np.float32)
            # 归一化向量
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-8)
            return result
        else:
            # 使用sentence-transformers
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
        if self.ollama_model_name:
            # ollama模式，使用缓存的维度
            return self._embedding_dim
        else:
            # sentence-transformers模式
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            else:
                # 通过测试向量获取维度
                test_embedding = self.embed("test")
                return len(test_embedding)
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        使Embedder对象可调用
        
        Args:
            texts: 单个文本字符串或文本列表
            
        Returns:
            numpy数组
        """
        return self.embed(texts)

