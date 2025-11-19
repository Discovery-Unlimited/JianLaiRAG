# Ollama 连接指南

本文档说明如何在本项目中连接和测试本地 Ollama 服务。

## 前置条件

1.  **安装 Ollama**: 确保本地已安装并运行 Ollama。
2.  **模型准备**: 确保已拉取所需的模型。本项目测试使用的模型为 `gpt-oss:120b-cloud`。
    *   如果未拉取，请在终端运行: `ollama pull gpt-oss:120b-cloud` (或者确认模型名称是否正确)。
3.  **服务状态**: 确保 Ollama 服务正在监听默认端口 `11434`。

## 连接配置

*   **API 地址**: `http://localhost:11434/api/generate`
*   **默认模型**: `gpt-oss:120b-cloud`

## 测试脚本

本项目提供了一个测试脚本 `tools/test_ollama.py` 用于验证连接。

### 使用方法

在项目根目录下运行以下命令：

```bash
python tools/test_ollama.py
```

### 交互说明

脚本运行后，会提示输入提示词（Prompt）：

```text
请输入提示词 (默认: '休谟是谁？'):
```

*   **直接回车**: 使用默认提示词 "休谟是谁？"。
*   **输入自定义内容**: 输入你想问的问题并回车。

### 预期输出

连接成功后，脚本将打印模型的回复以及耗时信息：

```text
Testing connection to Ollama at http://localhost:11434/api/generate with model 'gpt-oss:120b-cloud'...

Success! Response from Ollama:
--------------------------------------------------
(这里是模型的回复内容...)
--------------------------------------------------
Total duration: X.XX seconds
```

## 常见问题

*   **ConnectionError**: 无法连接到 Ollama。请检查 Ollama 是否已启动 (默认 `http://localhost:11434`)。
*   **HTTP Error**: 服务端返回错误。可能是模型不存在或参数错误。请检查模型名称是否正确。
