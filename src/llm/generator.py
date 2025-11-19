import requests
import json
import logging
from typing import Optional, Dict, Any, Generator

logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    LLM生成器，用于调用Ollama API生成文本。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM生成器。

        Args:
            config: LLM配置字典，包含 model_name, base_url 等。
        """
        self.model_name = config.get("model_name", "gpt-oss:120b-cloud")
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.api_url = f"{self.base_url}/api/generate"
        
        logger.info(f"Initialized LLMGenerator with model: {self.model_name} at {self.base_url}")

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        """
        根据提示词生成回复。

        Args:
            prompt: 提示词。
            stream: 是否流式输出。

        Returns:
            如果 stream=False，返回生成的完整字符串。
            如果 stream=True，返回一个生成器，逐个yield生成的token。
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(self.api_url, json=data, stream=stream)
            response.raise_for_status()

            if stream:
                return self._stream_response(response)
            else:
                return self._parse_response(response)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"

    def _parse_response(self, response: requests.Response) -> str:
        """解析非流式响应"""
        try:
            result = response.json()
            return result.get("response", "")
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON response")
            return ""

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """解析流式响应"""
        for line in response.iter_lines():
            if line:
                try:
                    json_obj = json.loads(line.decode('utf-8'))
                    token = json_obj.get("response", "")
                    if token:
                        yield token
                    if json_obj.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
