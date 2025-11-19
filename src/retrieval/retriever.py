import logging
from typing import List, Dict, Any, Optional
from ..knowledge_base.vector_store import VectorStore
from ..llm.embedder import Embedder

logger = logging.getLogger(__name__)

class Retriever:
    """
    检索器，负责从向量数据库中检索相关文档。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索器。

        Args:
            config: 配置字典，包含 embedding, vector_db 等配置。
        """
        self.config = config
        
        # 初始化嵌入模型
        embed_config = config.get("embedding", {})
        self.embedder = Embedder(
            model_name=embed_config.get("model_name", "ollama://bge-m3"),
            device=embed_config.get("device"),
            batch_size=embed_config.get("batch_size", 32)
        )
        
        # 初始化向量数据库
        db_config = config.get("vector_db", {})
        self.vector_store = VectorStore(
            persist_directory=db_config.get("path", "storage/vector_db"),
            collection_name=db_config.get("collection_name", "jianlai_novel")
        )
        
        logger.info("Initialized Retriever with Embedder and VectorStore")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索与查询相关的文档。

        Args:
            query: 查询文本。
            top_k: 返回结果数量。

        Returns:
            包含文档信息的字典列表。
        """
        logger.info(f"Retrieving top {top_k} documents for query: {query}")
        
        # 1. 生成查询向量
        query_embedding = self.embedder.embed(query)
        
        # 2. 在向量数据库中搜索
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
