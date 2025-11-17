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
from typing import Optional, Dict
import sys
import subprocess
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


def check_system_status() -> Dict[str, any]:
    """
    检测系统状态
    
    Returns:
        包含检测结果的字典
    """
    status = {
        'gpu_available': False,
        'gpu_info': None,
        'ollama_available': False,
        'ollama_gpu': False,
        'ollama_models': [],
        'sentence_transformers_available': False,
        'torch_cuda_available': False
    }
    
    print("=" * 60)
    print("系统环境检测")
    print("=" * 60)
    print()
    
    # 1. 检测PyTorch和CUDA
    print("1. 检测PyTorch和CUDA支持...")
    try:
        import torch
        status['torch_cuda_available'] = torch.cuda.is_available()
        if status['torch_cuda_available']:
            print(f"   ✅ PyTorch CUDA可用")
            print(f"   - CUDA版本: {torch.version.cuda}")
            print(f"   - GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                print(f"   - GPU {i}: {gpu_name} ({props.total_memory / (1024**3):.2f} GB)")
            status['gpu_available'] = True
            status['gpu_info'] = {
                'count': torch.cuda.device_count(),
                'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            }
        else:
            print(f"   ⚠️  PyTorch CUDA不可用，将使用CPU")
    except ImportError:
        print(f"   ❌ PyTorch未安装")
    
    # 2. 检测NVIDIA GPU（通过nvidia-smi）
    print("\n2. 检测NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ NVIDIA GPU驱动可用")
            status['gpu_available'] = True
            # 检查是否有ollama进程使用GPU
            if 'ollama' in result.stdout.lower():
                print("   ✅ 检测到Ollama进程可能在使用GPU")
                status['ollama_gpu'] = True
        else:
            print("   ⚠️  nvidia-smi执行失败")
    except FileNotFoundError:
        print("   ⚠️  nvidia-smi未找到，可能NVIDIA驱动未安装")
    except Exception as e:
        print(f"   ⚠️  检查GPU时出错: {e}")
    
    # 3. 检测Ollama服务
    print("\n3. 检测Ollama服务...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama服务正在运行")
            status['ollama_available'] = True
            
            # 检查已加载的模型
            try:
                ps_response = requests.get("http://localhost:11434/api/ps", timeout=5)
                if ps_response.status_code == 200:
                    models = ps_response.json().get("models", [])
                    if models:
                        print(f"   - 已加载模型数量: {len(models)}")
                        for model in models:
                            model_name = model.get('name', 'unknown')
                            size_vram = model.get('size_vram', 0)
                            if size_vram > 0:
                                print(f"     • {model_name} (VRAM: {size_vram / (1024**3):.2f} GB) ✅ GPU")
                                status['ollama_gpu'] = True
                            else:
                                print(f"     • {model_name} ⚠️  CPU")
                            status['ollama_models'].append(model_name)
                    else:
                        print("   - 当前没有加载的模型")
            except:
                pass
            
            # 检查bge-m3模型是否可用
            try:
                tags_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if tags_response.status_code == 200:
                    available_models = [m.get('name', '') for m in tags_response.json().get('models', [])]
                    if any('bge-m3' in m for m in available_models):
                        print("   ✅ bge-m3模型可用")
                    else:
                        print("   ⚠️  bge-m3模型未找到，需要运行: ollama pull bge-m3")
            except:
                pass
        else:
            print(f"   ❌ Ollama服务响应异常: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama服务未运行")
        print("   - 提示: 请确保Ollama已安装并正在运行")
    except ImportError:
        print("   ⚠️  requests库未安装，无法检测Ollama")
    except Exception as e:
        print(f"   ⚠️  检测Ollama时出错: {e}")
    
    # 4. 检测sentence-transformers
    print("\n4. 检测sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ sentence-transformers已安装")
        status['sentence_transformers_available'] = True
    except ImportError:
        print("   ❌ sentence-transformers未安装")
        print("   - 提示: 运行 pip install sentence-transformers")
    
    print("\n" + "=" * 60)
    return status


def interactive_model_selection(status: Dict[str, any]) -> Optional[str]:
    """
    交互式选择模型类型
    
    Args:
        status: 系统检测状态
        
    Returns:
        选择的模型类型，"ollama"或"st"，如果取消返回None
    """
    print("\n" + "=" * 60)
    print("模型选择")
    print("=" * 60)
    print()
    
    # 显示可用选项
    options = []
    
    # Ollama选项
    if status['ollama_available']:
        ollama_status = "✅ 可用"
        if status['ollama_gpu']:
            ollama_status += " (GPU)"
        else:
            ollama_status += " (CPU)"
        options.append(('ollama', f"Ollama API - {ollama_status}"))
    else:
        options.append(('ollama', "Ollama API - ❌ 不可用（服务未运行）"))
    
    # sentence-transformers选项
    if status['sentence_transformers_available']:
        st_status = "✅ 可用"
        if status['torch_cuda_available']:
            st_status += f" (GPU: {status['gpu_info']['devices'][0] if status['gpu_info'] else 'N/A'})"
        else:
            st_status += " (CPU)"
        options.append(('st', f"sentence-transformers - {st_status}"))
    else:
        options.append(('st', "sentence-transformers - ❌ 不可用（未安装）"))
    
    # 显示选项
    print("请选择要使用的嵌入模型：")
    print()
    for i, (value, desc) in enumerate(options, 1):
        print(f"  {i}. {desc}")
    print(f"  0. 退出")
    print()
    
    # 获取用户输入
    while True:
        try:
            choice = input("请输入选项编号 (0-2): ").strip()
            
            if choice == '0':
                print("已取消操作")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1][0]
                
                # 检查是否可用
                if selected == 'ollama' and not status['ollama_available']:
                    print("❌ Ollama不可用，请先启动Ollama服务")
                    continue
                elif selected == 'st' and not status['sentence_transformers_available']:
                    print("❌ sentence-transformers不可用，请先安装: pip install sentence-transformers")
                    continue
                
                print(f"\n✅ 已选择: {options[choice_num - 1][1]}")
                return selected
            else:
                print(f"❌ 无效选项，请输入 0-{len(options)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消操作")
            return None


def build_knowledge_base(
    config_path: str = "config/settings.yaml",
    data_path: Optional[str] = None,
    reset: bool = False,
    model_type: Optional[str] = None
):
    """
    构建知识库
    
    Args:
        config_path: 配置文件路径
        data_path: 数据源路径（如果为None，使用配置文件中的路径）
        reset: 是否重置向量数据库
        model_type: 模型类型，"ollama"或"st"（sentence-transformers），如果为None则使用配置文件
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
    
    # 根据命令行参数或配置文件确定模型类型
    if model_type:
        if model_type.lower() == "ollama":
            model_name = "ollama://bge-m3"
            logger.info("使用命令行参数: 选择Ollama模型")
        elif model_type.lower() in ["st", "sentence-transformers", "sentence_transformers"]:
            model_name = "BAAI/bge-m3"
            logger.info("使用命令行参数: 选择sentence-transformers模型")
        else:
            logger.warning(f"未知的模型类型: {model_type}，使用配置文件中的设置")
            model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
    else:
        model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
        logger.info(f"使用配置文件中的模型: {model_name}")

    # sentence-transformers模型优先尝试使用本地缓存目录
    if not model_name.startswith("ollama://"):
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
  # 交互式选择模型（默认）
  python build_kb.py
  
  # 直接指定使用Ollama模型（跳过交互）
  python build_kb.py --model-type ollama
  
  # 直接指定使用sentence-transformers模型（跳过交互）
  python build_kb.py --model-type st
  
  # 使用自定义数据路径和重置数据库
  python build_kb.py --data data/raw --reset --model-type ollama
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
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['ollama', 'st', 'sentence-transformers', 'sentence_transformers'],
        default=None,
        help='模型类型: "ollama" 使用Ollama API, "st" 使用sentence-transformers（如果未指定，将进行交互式选择）'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='跳过系统检测（仅在指定--model-type时有效）'
    )
    
    args = parser.parse_args()
    
    try:
        # 如果未指定模型类型，进行检测和交互式选择
        model_type = args.model_type
        if model_type is None:
            # 进行系统检测
            status = check_system_status()
            
            # 交互式选择
            model_type = interactive_model_selection(status)
            
            if model_type is None:
                print("操作已取消")
                sys.exit(0)
        elif not args.skip_check:
            # 即使指定了模型类型，也进行快速检测以显示状态
            print("=" * 60)
            print("快速检测")
            print("=" * 60)
            status = check_system_status()
            print()
        
        # 构建知识库
        build_knowledge_base(
            config_path=args.config,
            data_path=args.data,
            reset=args.reset,
            model_type=model_type
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

