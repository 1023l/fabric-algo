"""LLM 客户端：OpenAI 兼容接口（DeepSeek / Qwen / OpenAI / 本地 vLLM 等）。

通过环境变量（或项目根 .env）配置：
  QA_LLM_API_KEY   - API Key
  QA_LLM_BASE_URL  - 接口地址（默认 https://api.deepseek.com）
  QA_LLM_MODEL     - 模型名（默认 deepseek-chat）
  QA_LLM_MAX_STEPS - 最大工具调用轮次（默认 10，防止无限循环）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    max_steps: int = 10
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        api_key = os.getenv("QA_LLM_API_KEY", "")
        if not api_key:
            raise ValueError(
                "QA_LLM_API_KEY 未设置。请在 .env 中配置 LLM API Key（QA_LLM_API_KEY/...）。"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("QA_LLM_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("QA_LLM_MODEL", "deepseek-chat"),
            max_steps=int(os.getenv("QA_LLM_MAX_STEPS", "10")),
            temperature=float(os.getenv("QA_LLM_TEMPERATURE", "0.0")),
        )


class LLMClient:
    """OpenAI 兼容 LLM 客户端，支持 function calling。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """调用 LLM，返回完整 response 对象（调用方解析 tool_calls 决定下一步）。"""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)
