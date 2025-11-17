#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Ollama GPU使用情况
"""

import requests
import subprocess
import sys

def check_ollama_gpu():
    """检查Ollama是否使用GPU"""
    print("=" * 60)
    print("Ollama GPU使用情况检查")
    print("=" * 60)
    print()
    
    # 1. 检查Ollama服务是否运行
    print("1. 检查Ollama服务...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama服务正在运行")
        else:
            print(f"   ❌ Ollama服务响应异常: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ 无法连接到Ollama服务")
        print("   请确保Ollama已安装并正在运行")
        return
    
    # 2. 检查已加载的模型
    print("\n2. 检查已加载的模型...")
    try:
        ps_response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if ps_response.status_code == 200:
            models = ps_response.json().get("models", [])
            if models:
                for model in models:
                    print(f"   模型: {model.get('name', 'unknown')}")
                    size_vram = model.get('size_vram', 0)
                    if size_vram > 0:
                        print(f"   - VRAM使用: {size_vram / (1024**3):.2f} GB")
                        print(f"   ✅ 模型正在使用GPU")
                    else:
                        print(f"   - 大小: {model.get('size', 0) / (1024**3):.2f} GB")
                        print(f"   ⚠️  模型可能在使用CPU")
            else:
                print("   当前没有加载的模型")
    except Exception as e:
        print(f"   ⚠️  无法获取模型信息: {e}")
    
    # 3. 检查NVIDIA GPU
    print("\n3. 检查NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ NVIDIA GPU可用")
            # 检查是否有ollama进程使用GPU
            output_lower = result.stdout.lower()
            if 'ollama' in output_lower:
                print("   ✅ 检测到Ollama进程使用GPU")
                # 提取相关信息
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'ollama' in line.lower():
                        print(f"   - {line.strip()}")
            else:
                print("   ⚠️  未检测到Ollama进程使用GPU")
                print("   可能原因：")
                print("   1. Ollama服务启动时未设置GPU环境变量")
                print("   2. 需要重启Ollama服务")
        else:
            print("   ❌ nvidia-smi执行失败")
    except FileNotFoundError:
        print("   ❌ nvidia-smi未找到，可能NVIDIA驱动未安装")
    except Exception as e:
        print(f"   ⚠️  检查GPU时出错: {e}")
    
    # 4. 提供解决方案
    print("\n" + "=" * 60)
    print("解决方案")
    print("=" * 60)
    print("\n要让Ollama使用GPU，请按以下步骤操作：\n")
    print("Windows PowerShell:")
    print("  1. 停止Ollama服务（任务管理器结束ollama进程）")
    print("  2. 设置环境变量:")
    print("     $env:CUDA_VISIBLE_DEVICES='0'")
    print("  3. 重新启动Ollama服务")
    print()
    print("或者使用系统环境变量（永久设置）：")
    print("  1. 右键'此电脑' -> 属性 -> 高级系统设置")
    print("  2. 环境变量 -> 系统变量 -> 新建")
    print("  3. 变量名: CUDA_VISIBLE_DEVICES")
    print("  4. 变量值: 0")
    print("  5. 重启Ollama服务")
    print()
    print("Linux/macOS:")
    print("  export CUDA_VISIBLE_DEVICES=0")
    print("  ollama serve")
    print("=" * 60)

if __name__ == "__main__":
    check_ollama_gpu()

