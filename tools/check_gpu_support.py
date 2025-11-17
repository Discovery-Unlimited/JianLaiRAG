#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GPU支持情况的诊断脚本
"""

import sys

print("=" * 60)
print("GPU支持诊断工具")
print("=" * 60)
print()

# 1. 检查PyTorch是否安装
print("1. 检查PyTorch安装情况...")
try:
    import torch
    print(f"   ✅ PyTorch已安装")
    print(f"   - 版本: {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch未安装")
    print("   - 需要安装PyTorch才能使用GPU")
    sys.exit(1)

# 2. 检查CUDA是否可用
print("\n2. 检查CUDA支持...")
if torch.cuda.is_available():
    print("   ✅ CUDA可用")
    print(f"   - CUDA版本: {torch.version.cuda}")
    print(f"   - cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"   - GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   - 名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   - 总显存: {props.total_memory / (1024**3):.2f} GB")
        print(f"   - 计算能力: {props.major}.{props.minor}")
else:
    print("   ❌ CUDA不可用")
    print("\n   可能的原因：")
    print("   1. PyTorch安装的是CPU版本（不是GPU版本）")
    print("   2. NVIDIA驱动未安装或版本过旧")
    print("   3. CUDA工具包未安装或版本不匹配")
    
    # 检查是否有NVIDIA驱动
    print("\n   检查NVIDIA驱动...")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ nvidia-smi可用，NVIDIA驱动已安装")
            print("   - 输出前几行：")
            for line in result.stdout.split('\n')[:5]:
                if line.strip():
                    print(f"     {line}")
        else:
            print("   ❌ nvidia-smi不可用")
    except FileNotFoundError:
        print("   ❌ nvidia-smi未找到，可能NVIDIA驱动未安装")
    except Exception as e:
        print(f"   ⚠️  检查nvidia-smi时出错: {e}")

# 3. 检查当前PyTorch是否支持CUDA
print("\n3. 检查PyTorch CUDA支持...")
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f"   ✅ PyTorch编译时支持CUDA: {torch.version.cuda}")
else:
    print("   ❌ PyTorch是CPU版本，不支持CUDA")
    print("   - 需要重新安装PyTorch GPU版本")

# 4. 测试GPU计算
print("\n4. 测试GPU计算...")
if torch.cuda.is_available():
    try:
        # 创建一个简单的tensor测试
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU计算测试成功")
        print(f"   - 测试tensor设备: {z.device}")
    except Exception as e:
        print(f"   ❌ GPU计算测试失败: {e}")
else:
    print("   ⏭️  跳过（CUDA不可用）")

# 5. 提供解决方案
print("\n" + "=" * 60)
print("解决方案")
print("=" * 60)

if not torch.cuda.is_available():
    print("\n要启用GPU支持，请按以下步骤操作：\n")
    
    print("方法1：安装PyTorch GPU版本（推荐）")
    print("-" * 60)
    print("1. 首先检查您的CUDA版本：")
    print("   nvidia-smi")
    print()
    print("2. 根据CUDA版本安装对应的PyTorch：")
    print()
    print("   # 对于CUDA 11.8（MX350推荐）：")
    print("   pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("   # 对于CUDA 12.1：")
    print("   pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("   # 对于CUDA 13：")
    print("   pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu130")
    print()
    print("   # 如果不知道CUDA版本，可以尝试：")
    print("   pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu118")
    print()
    
    print("方法2：使用conda安装（如果有conda）")
    print("-" * 60)
    print("   conda install pytorch torchvision  pytorch-cuda=11.8 -c pytorch -c nvidia")
    print()

else:
    print("\n✅ GPU已可用！")
    print("您可以直接使用GPU模式：")
    print("   python build_rag_knowledge_base.py --device cuda")
    print()

print("=" * 60)

