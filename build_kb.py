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
from typing import Optional, List, Any
import sys
import pickle
import hashlib

from src.knowledge_base import DocumentLoader, TextSplitter, VectorStore
from src.llm import Embedder


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


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"配置文件为空或格式错误: {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"YAML解析错误: {e}")


def resolve_model_path(
    model_name: str,
    local_model_dir: Optional[str],
    logger: logging.Logger
) -> str:
    """
    解析模型路径：优先使用本地路径，否则使用HuggingFace模型名
    
    Args:
        model_name: HuggingFace模型名称（如 "BAAI/bge-m3"）
        local_model_dir: 本地模型目录路径（如 "models/bge-m3"）
        logger: 日志记录器
        
    Returns:
        解析后的模型路径或HuggingFace模型名称
    """
    # 优先检查本地模型目录（这是明确的本地路径）
    if local_model_dir:
        local_dir = Path(local_model_dir)
        if local_dir.exists():
            resolved_path = str(local_dir.resolve())
            logger.info(f"检测到本地模型路径: {resolved_path}")
            return resolved_path
    
    # 如果本地模型不存在，检查 model_name 是否可能是本地路径
    # （兼容用户直接指定本地路径的情况）
    model_path = Path(model_name)
    if model_path.exists():
        resolved_path = str(model_path.resolve())
        logger.info(f"检测到本地模型路径: {resolved_path}")
        return resolved_path
    
    # 如果都不存在，使用 model_name 作为 HuggingFace 模型名称
    logger.info(f"未检测到本地模型目录，将使用 HuggingFace 模型: {model_name}")
    return model_name


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
    config_path: str = "config/settings.yaml",
    data_path: Optional[str] = None,
    reset: bool = False
) -> bool:
    """
    构建知识库
    
    Args:
        config_path: 配置文件路径
        data_path: 数据源路径（如果为None，使用配置文件中的路径）
        reset: 是否重置向量数据库
        
    Returns:
        是否成功构建
    """
    # 先尝试读取配置文件来获取日志设置（如果配置文件存在）
    # 这样第一次的日志也能写入文件
    log_level = "INFO"
    log_file = None
    try:
        # 静默读取配置文件，获取日志设置
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                temp_config = yaml.safe_load(f)
            if temp_config:
                logging_config = temp_config.get('logging', {})
                log_level = logging_config.get('level', 'INFO')
                log_file = logging_config.get('file')
    except Exception:
        # 如果读取失败，使用默认设置
        pass
    
    # 使用获取到的日志设置初始化日志（如果配置文件存在，日志会写入文件）
    temp_logger = setup_logging(log_level, log_file)
    temp_logger.info(f"加载配置文件: {config_path}")
    
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        temp_logger.error(f"加载配置失败: {e}")
        return False
    
    # 获取配置项
    embedding_config = config.get('embedding', {})
    vector_db_config = config.get('vector_db', {})
    data_config = config.get('data', {})
    text_splitter_config = config.get('text_splitter', {})
    logging_config = config.get('logging', {})
    
    # 根据配置重新设置日志（确保配置正确，并更新日志级别）
    log_level = logging_config.get('level', 'INFO')
    log_file = logging_config.get('file')
    logger = setup_logging(log_level, log_file, force=True)
    
    # 数据源路径
    if data_path is None:
        data_path = data_config.get('raw_path', 'data/raw')
    
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.error(f"数据路径不存在: {data_path}")
        return False
    
    # 根据配置确定模型路径：models目录优先，缺失时走HuggingFace
    model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
    local_model_dir = embedding_config.get('local_model_dir', 'models/bge-m3')
    model_name = resolve_model_path(model_name, local_model_dir, logger)
    
    logger.info("=" * 60)
    logger.info("开始构建知识库")
    logger.info("=" * 60)
    
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
            chunk_size=text_splitter_config.get('chunk_size', 500),
            chunk_overlap=text_splitter_config.get('chunk_overlap', 50),
            split_by_chapter=text_splitter_config.get('split_by_chapter', True),
            min_chunk_size=text_splitter_config.get('min_chunk_size', 100)
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
    cache_dir = Path(vector_db_config.get('cache_dir', 'storage/cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{vector_db_config.get('collection_name', 'jianlai_novel')}_embeddings.pkl"

    chunks_signature = build_chunks_signature(chunks)
    
    # 3. 初始化嵌入模型
    logger.info("\n[步骤 3/5] 初始化嵌入模型...")
    try:
        embedder = Embedder(
            model_name=model_name,
            local_model_dir=embedding_config.get('local_model_dir', 'models/bge-m3'),
            device=embedding_config.get('device', 'cuda'),
            batch_size=embedding_config.get('batch_size', 32),
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
        
        return True
    except Exception as e:
        logger.error(f"存储到向量数据库失败: {e}")
        return False


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
        success = build_knowledge_base(
            config_path=args.config,
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

