#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库构建脚本
整合所有模块，实现完整的知识库构建流程
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Optional
import sys
import pickle
import hashlib

from src.knowledge_base import DocumentLoader, TextSplitter, VectorStore
from src.llm import Embedder


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config



def build_knowledge_base(
    config_path: str = "config/settings.yaml",
    data_path: Optional[str] = None,
    reset: bool = False
):
    """
    构建知识库
    
    Args:
        config_path: 配置文件路径
        data_path: 数据源路径（如果为None，使用配置文件中的路径）
        reset: 是否重置向量数据库
    """
    logger = setup_logging()
    
    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 获取配置项
    embedding_config = config.get('embedding', {})
    vector_db_config = config.get('vector_db', {})
    data_config = config.get('data', {})
    text_splitter_config = config.get('text_splitter', {})
    logging_config = config.get('logging', {})
    
    # 设置日志
    log_level = logging_config.get('level', 'INFO')
    log_file = logging_config.get('file')
    logger = setup_logging(log_level, log_file)
    
    # 数据源路径
    if data_path is None:
        data_path = data_config.get('raw_path', 'data/raw')
    
    # 根据配置确定模型路径：models目录优先，缺失时走HuggingFace
    model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
    local_model_dir = embedding_config.get('local_model_dir', 'models/bge-m3')
    local_model_path = Path(model_name)

    if local_model_path.exists():
        model_name = str(local_model_path.resolve())
        logger.info(f"检测到本地模型路径: {model_name}")
    elif local_model_dir and Path(local_model_dir).exists():
        model_name = str(Path(local_model_dir).resolve())
        logger.info(f"未找到指定模型，改为加载本地缓存模型: {model_name}")
    else:
        logger.info("未检测到本地模型目录，将通过HuggingFace自动下载")
    
    logger.info("=" * 60)
    logger.info("开始构建知识库")
    logger.info("=" * 60)
    
    # 1. 加载文档
    logger.info("\n[步骤 1/5] 加载文档...")
    loader = DocumentLoader()
    documents = loader.load_directory(data_path)
    if not documents:
        logger.error("未找到任何文档，请检查数据路径")
        return
    logger.info(f"成功加载 {len(documents)} 个文档")
    
    # 2. 分割文本
    logger.info("\n[步骤 2/5] 分割文本...")
    splitter = TextSplitter(
        chunk_size=text_splitter_config.get('chunk_size', 500),
        chunk_overlap=text_splitter_config.get('chunk_overlap', 50),
        split_by_chapter=text_splitter_config.get('split_by_chapter', True),
        min_chunk_size=text_splitter_config.get('min_chunk_size', 100)
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"文本分割完成，共生成 {len(chunks)} 个文本块")

    # 缓存目录与文件（用于在步骤5失败后复用嵌入结果）
    cache_dir = Path(vector_db_config.get('cache_dir', 'storage/cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{vector_db_config.get('collection_name', 'jianlai_novel')}_embeddings.pkl"

    def build_chunks_signature(chunks_list):
        hasher = hashlib.md5()
        for chunk in chunks_list:
            part = f"{chunk.metadata.get('filename', '')}-{chunk.metadata.get('chapter', '')}-{chunk.metadata.get('chunk_idx', 0)}-{len(chunk.text)}"
            hasher.update(part.encode('utf-8', errors='ignore'))
        return hasher.hexdigest()

    chunks_signature = build_chunks_signature(chunks)
    loaded_from_cache = False
    
    # 3. 初始化嵌入模型
    logger.info("\n[步骤 3/5] 初始化嵌入模型...")
    embedder = Embedder(
        model_name=model_name,
        local_model_dir=embedding_config.get('local_model_dir', 'models/bge-m3'),
        device=embedding_config.get('device', 'cuda'),
        batch_size=embedding_config.get('batch_size', 32)
    )
    logger.info(f"嵌入模型初始化完成，向量维度: {embedder.dimension}")
    
    # 4. 向量化文本块（支持缓存）
    cached_embeddings = None
    if cache_file.exists() and not reset:
        try:
            with cache_file.open('rb') as f:
                cache_data = pickle.load(f)
            if cache_data.get('signature') == chunks_signature:
                cached_embeddings = cache_data.get('embeddings')
                loaded_from_cache = True
                logger.info(f"检测到缓存嵌入文件，跳过重新向量化: {cache_file}")
            else:
                logger.info("缓存签名与当前文本不匹配，将重新计算嵌入")
        except Exception as e:
            logger.warning(f"加载嵌入缓存失败，将重新计算: {e}")
    
    if cached_embeddings is not None:
        embeddings = cached_embeddings
    else:
        logger.info("\n[步骤 4/5] 向量化文本块...")
        texts = [chunk.text for chunk in chunks]
        logger.info(f"开始向量化 {len(texts)} 个文本块...")
        embeddings = embedder.embed_batch(texts, show_progress=True)
        logger.info(f"向量化完成，生成 {len(embeddings)} 个向量")

        try:
            with cache_file.open('wb') as f:
                pickle.dump({
                    'signature': chunks_signature,
                    'embeddings': embeddings
                }, f)
            logger.info(f"嵌入结果已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"写入嵌入缓存失败: {e}")
    
    if loaded_from_cache:
        logger.info("使用缓存嵌入完成步骤 4/5")
    
    # 5. 存储到向量数据库
    logger.info("\n[步骤 5/5] 存储到向量数据库...")
    vector_store = VectorStore(
        persist_directory=vector_db_config.get('path', 'storage/vector_db'),
        collection_name=vector_db_config.get('collection_name', 'jianlai_novel')
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='构建RAG知识库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_kb.py
  # 使用自定义数据路径和重置数据库
  python build_kb.py --data data/raw --reset
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='配置文件路径（默认: config/settings.yaml）'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据源路径（如果指定，将覆盖配置文件中的设置）'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='重置向量数据库（清空现有数据）'
    )
    
    args = parser.parse_args()
    
    try:
        build_knowledge_base(
            config_path=args.config,
            data_path=args.data,
            reset=args.reset
        )
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

