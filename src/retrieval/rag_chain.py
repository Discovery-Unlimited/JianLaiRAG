import logging
from typing import Dict, Any, Optional, Generator
from .retriever import Retriever
from ..llm.generator import LLMGenerator

logger = logging.getLogger(__name__)

class RAGChain:
    """
    RAG链，负责协调检索和生成过程。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG链。

        Args:
            config: 系统配置字典。
        """
        self.config = config
        self.retriever = Retriever(config)
        self.generator = LLMGenerator(config.get("llm", {}))
        
        logger.info("Initialized RAGChain")

    def answer(self, question: str, top_k: int = 5, stream: bool = True) -> Generator[str, None, None] | str:
        """
        回答问题。

        Args:
            question: 用户问题。
            top_k: 检索的文档数量。
            stream: 是否流式输出。

        Returns:
            生成的答案（字符串或生成器）。
        """
        logger.info(f"Processing question: {question}")
        
        # 1. 检索相关文档
        documents = self.retriever.retrieve(question, top_k=top_k)
        
        # 2. 构造提示词
        context = self._format_context(documents)
        prompt = self._construct_prompt(question, context)
        
        # 3. 生成答案
        logger.info("Generating answer...")
        return self.generator.generate(prompt, stream=stream)

    def _format_context(self, documents: list) -> str:
        """格式化检索到的文档作为上下文"""
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.get('metadata', {}).get('filename', 'unknown')
            text = doc.get('document', '').strip()
            context_parts.append(f"文档 {i+1} (来源: {source}):\n{text}")
        
        return "\n\n".join(context_parts)

    def _construct_prompt(self, question: str, context: str) -> str:
        """构造提示词"""
        return f"""你是一个精通《剑来》小说的AI助手。请基于以下检索到的参考文档回答用户的问题。
如果参考文档中没有相关信息，请如实告知，不要编造。

参考文档：
{context}

用户问题：{question}

请回答："""
