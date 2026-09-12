"""Agent 运行时：ReAct 循环（基于 function calling）。

核心循环：
  1. 把 user 问题 + system prompt + tools 发给 LLM
  2. LLM 返回：要么是最终回答，要么是 tool_calls
  3. 如果是 tool_calls → 执行对应工具 → 把结果加入 messages → 回到第 1 步
  4. 如果是最终回答 → 返回结果
  5. 超过 max_steps 仍未完成 → 强制收尾

每一步的轨迹（思考 + 工具调用 + 结果）都被记录，用于前端可视化。
（移植自 bi-cli agent/runtime.py，逻辑不变。）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from .llm import LLMClient
from .prompts import build_system_prompt
from .tools import TOOL_HANDLERS, TOOL_SCHEMAS


@dataclass
class Step:
    """一轮 ReAct 循环的记录。"""
    step_num: int
    thought: str = ""          # LLM 的思考/回答文本
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 工具调用
    tool_results: list[dict[str, Any]] = field(default_factory=list)  # 工具返回
    is_final: bool = False     # 是否是最终回答


@dataclass
class AgentResult:
    """Agent 完整运行结果。"""
    answer: str
    steps: list[Step] = field(default_factory=list)
    total_tokens: int = 0

    def to_trace(self) -> list[dict[str, Any]]:
        """转为可序列化的轨迹 JSON。"""
        return [
            {
                "step": s.step_num,
                "thought": s.thought,
                "tool_calls": s.tool_calls,
                "tool_results": s.tool_results,
                "is_final": s.is_final,
            }
            for s in self.steps
        ]


class AgentRuntime:
    """ReAct agent 运行时。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.tools = TOOL_SCHEMAS
        self.tool_handlers = TOOL_HANDLERS

    def run(self, question: str) -> AgentResult:
        """同步执行：问一个问题，返回完整结果。"""
        result = AgentResult(answer="")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ]

        for step_num in range(1, self.llm.config.max_steps + 1):
            step = Step(step_num=step_num)

            try:
                response = self.llm.chat(messages, tools=self.tools)
            except Exception as e:
                step.thought = f"LLM 调用失败：{e}"
                step.is_final = True
                result.steps.append(step)
                result.answer = f"Agent 运行出错：{e}"
                break

            choice = response.choices[0]
            msg = choice.message

            # 记录 LLM 思考
            step.thought = msg.content or ""

            # 检查是否有工具调用
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_info = {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    step.tool_calls.append(call_info)

                # 把 assistant 的 tool_calls 消息加入历史
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # 逐个执行工具
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    handler = self.tool_handlers.get(tool_name)
                    if handler is None:
                        tool_result = f"未知工具：{tool_name}"
                    else:
                        try:
                            tool_result = handler(**args)
                        except Exception as e:
                            tool_result = f"工具执行异常：{e}"

                    step.tool_results.append({
                        "name": tool_name,
                        "result": tool_result,
                    })

                    # 把工具结果加入消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

                result.steps.append(step)
                # 继续下一轮，让 LLM 看到工具结果后决定下一步
                continue

            # 没有工具调用 = 最终回答
            step.is_final = True
            result.steps.append(step)
            result.answer = msg.content or ""
            break

        else:
            # 超过最大轮次
            result.answer = "Agent 达到最大工具调用轮次，未能完成分析。"
            result.steps.append(Step(
                step_num=self.llm.config.max_steps,
                thought="达到最大轮次限制",
                is_final=True,
            ))

        # 估算 token 用量
        try:
            result.total_tokens = response.usage.total_tokens
        except (AttributeError, NameError):
            pass

        return result

    def run_stream(self, question: str) -> Iterator[dict[str, Any]]:
        """流式执行：逐步 yield 每一步的轨迹，供前端实时展示。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ]

        for step_num in range(1, self.llm.config.max_steps + 1):
            step_data: dict[str, Any] = {"step": step_num, "tool_calls": [], "tool_results": []}

            try:
                response = self.llm.chat(messages, tools=self.tools)
            except Exception as e:
                step_data["thought"] = f"LLM 调用失败：{e}"
                step_data["is_final"] = True
                step_data["answer"] = f"Agent 运行出错：{e}"
                yield step_data
                return

            choice = response.choices[0]
            msg = choice.message
            step_data["thought"] = msg.content or ""

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step_data["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    handler = self.tool_handlers.get(tool_name)
                    if handler is None:
                        tool_result = f"未知工具：{tool_name}"
                    else:
                        try:
                            tool_result = handler(**args)
                        except Exception as e:
                            tool_result = f"工具执行异常：{e}"

                    step_data["tool_results"].append({
                        "name": tool_name,
                        "result": tool_result,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

                yield step_data
                continue

            # 最终回答
            step_data["is_final"] = True
            step_data["answer"] = msg.content or ""
            yield step_data
            return

        # 超时
        yield {
            "step": self.llm.config.max_steps,
            "thought": "达到最大轮次限制",
            "is_final": True,
            "answer": "Agent 达到最大工具调用轮次，未能完成分析。",
        }
