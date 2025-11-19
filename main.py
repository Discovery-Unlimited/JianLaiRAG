#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剑来RAG问答系统主程序
"""

import argparse
import yaml
import logging
import sys
from typing import Dict, Any
from src.retrieval.rag_chain import RAGChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='剑来RAG问答系统')
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='禁用流式输出'
    )
    args = parser.parse_args()

    # 1. 加载配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # 2. 初始化RAG链
    try:
        rag_chain = RAGChain(config)
    except Exception as e:
        logger.error(f"Failed to initialize RAG Chain: {e}")
        sys.exit(1)

    # 3. 交互式循环
    print("\n" + "="*50)
    print("欢迎使用剑来RAG问答系统！")
    print("输入 'quit', 'exit' 或 'q' 退出")
    print("="*50 + "\n")

    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            print("\n正在思考...")
            
            response = rag_chain.answer(
                question, 
                stream=not args.no_stream
            )

            print("\n回答:")
            print("-" * 50)
            
            if not args.no_stream:
                # 流式输出
                for token in response:
                    print(token, end='', flush=True)
                print()
            else:
                # 一次性输出
                print(response)
                
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n用户中断操作")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
