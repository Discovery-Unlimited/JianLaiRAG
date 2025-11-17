# 开发环境说明

## requirements.txt

```
# 向量数据库
chromadb>=0.4.0
# 嵌入模型
sentence-transformers>=2.2.0
# 配置文件解析
pyyaml>=6.0
# 深度学习框架（用于嵌入模型）
torch>=2.0.0
# 数值计算
numpy>=1.24.0
# 网络请求
requests>=2.28.0
```

## 手动安装

```
pip install chromadb --no-build-isolation
pip install -r requirements.txt
```