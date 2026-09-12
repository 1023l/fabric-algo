"""工具层：产线数据查询能力注册为 LLM 可调用的 tools（function calling）。

（移植自 bi-cli agent/tools.py，数据层从 StarRocks bi_cli 换成本仓库 qa_agent.store 的 SQLite。）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from . import store

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
FACTS_PATH = KNOWLEDGE_DIR / "facts.yaml"

# ---------------------------------------------------------------------------
# 表注册表（给 LLM 的元数据；查询白名单在 store.py 内强制）
# ---------------------------------------------------------------------------

TABLES: dict[str, dict[str, Any]] = {
    "pieces": {
        "kind": "fact",  # 过线事实明细
        "description": "布片过线明细：每个布片穿过计数线时一条记录",
        "fields": {
            "id": "记录ID",
            "session_id": "检测会话ID（一次启动检测=一个会话）",
            "ts": "过线时间（本地时间 ISO，如 2026-09-12T10:23:45）",
            "track_id": "跟踪ID（会话内布片编号）",
            "direction": "过线方向：down=正向(下线) / up=反向(回退)",
            "conf": "检测置信度",
            "text": "OCR 原文（如 '30-30TR L'，可能为 NULL）",
            "shoe_size": "鞋码（text 中 '-' 前数字，可为 NULL）",
            "lr_flag": "L/R 标记：'L' / 'R' / 'L/R'（可为 NULL）",
        },
        "filters": ["ts", "ts_start/ts_end", "direction", "shoe_size", "session_id"],
        "groupable": ["direction", "shoe_size", "lr_flag", "session_id"],
    },
    "v_hourly": {
        "kind": "view",  # 时段聚合视图
        "description": "按小时聚合视图：每小时正向/反向/累计/L/R 计数（推荐汇总类问题优先用）",
        "fields": {
            "hour": "小时（如 '2026-09-12T10'）",
            "down_cnt": "该小时正向过线数",
            "up_cnt": "该小时反向过线数",
            "net_cnt": "累计（正向-反向）",
            "l_cnt": "含 L 的布片数",
            "r_cnt": "含 R 的布片数",
            "size_kinds": "该小时出现过的鞋码种类数",
        },
        "filters": ["hour（字符串前缀比较，如 hour >= '2026-09-12T00'）"],
        "groupable": [],
    },
    "v_sizes": {
        "kind": "view",  # 鞋码分布视图
        "description": "鞋码分布视图：每个鞋码的正向/反向/总数（鞋码分布类问题优先用）",
        "fields": {
            "shoe_size": "鞋码",
            "down_cnt": "正向数",
            "up_cnt": "反向数",
            "total": "总数",
        },
        "filters": ["shoe_size"],
        "groupable": [],
    },
}


# ---------------------------------------------------------------------------
# 工具执行函数
# ---------------------------------------------------------------------------

def read_document(path: str) -> str:
    """读取知识层文档（表文档、facts.yaml 等）。"""
    target = (KNOWLEDGE_DIR / path).resolve()
    if not target.is_relative_to(KNOWLEDGE_DIR):
        return f"错误：路径越界，只允许读知识层目录内文件：{path}"
    if not target.exists():
        return f"文件不存在：{path}"
    content = target.read_text(encoding="utf-8")
    max_chars = 8000
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... (已截断，完整文件共 {len(content)} 字符)"
    return content


def list_tables() -> str:
    """列出所有可查询的表/视图及其字段说明。"""
    return json.dumps(TABLES, ensure_ascii=False, indent=2)


def query_data(
    table: str,
    fields: str = "",
    ts_start: str = "",
    ts_end: str = "",
    direction: str = "",
    shoe_size: int | None = None,
    session_id: str = "",
    group_by: str = "",
    order_by: str = "",
    limit: int = 200,
) -> str:
    """执行结构化查询（pieces 表），返回真实数据。"""
    if table != "pieces":
        return (f"表 '{table}' 不支持结构化查询。可查询：pieces。"
                f"汇总类问题请改用 run_sql 查 v_hourly / v_sizes 视图。")
    result = store.query_pieces(
        fields=[f.strip() for f in fields.split(",") if f.strip()] or None,
        ts_start=ts_start or None,
        ts_end=ts_end or None,
        direction=direction or None,
        shoe_size=shoe_size,
        session_id=session_id or None,
        group_by=[g.strip() for g in group_by.split(",") if g.strip()] or None,
        order_by=order_by or None,
        limit=limit,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def run_sql(sql: str, limit: int = 200) -> str:
    """受限只读 SQL（用于 v_hourly / v_sizes 视图和聚合兜底）。仅允许 SELECT。"""
    return json.dumps(store.raw_sql(sql, limit=limit), ensure_ascii=False, indent=2)


def get_facts(table: str = "") -> str:
    """查询知识层事实账本（facts.yaml），获取口径定义、数据陷阱等。"""
    try:
        data = yaml.safe_load(FACTS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return f"加载 facts.yaml 失败：{e}"

    facts = data.get("facts", [])
    if table:
        facts = [f for f in facts if table in (f.get("tables") or [])]
    return json.dumps(facts, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 工具注册表（给 LLM 的 function calling schema）
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "读取知识层文档（表文档、facts.yaml）。"
                "路径相对于知识层目录，如 'tables/pieces.md'、'facts.yaml'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，如 'tables/pieces.md'"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出所有可查询的表/视图，含字段名、含义、过滤维度。不需要参数。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "结构化查询 pieces 表（过线明细），返回真实数据。"
                "适合明细追溯、按方向/鞋码分组计数。"
                "汇总类问题（小时趋势、鞋码分布）优先用 run_sql 查视图。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "固定填 pieces"},
                    "fields": {"type": "string", "description": "字段逗号分隔，默认 'ts,direction,shoe_size,lr_flag,text'"},
                    "ts_start": {"type": "string", "description": "起始时间（含），如 '2026-09-12T00:00:00'"},
                    "ts_end": {"type": "string", "description": "结束时间（含），如 '2026-09-12T23:59:59'"},
                    "direction": {"type": "string", "enum": ["down", "up"], "description": "过线方向过滤，可空"},
                    "shoe_size": {"type": "integer", "description": "鞋码过滤，可空"},
                    "session_id": {"type": "string", "description": "会话过滤，可空"},
                    "group_by": {"type": "string", "description": "分组字段逗号分隔（分组后自动带 cnt 计数）"},
                    "order_by": {"type": "string", "description": "排序字段，前缀 - 表示降序，如 '-ts'"},
                    "limit": {"type": "integer", "description": "返回行数上限，默认 200"},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "受限只读 SQL 查询（仅 SELECT），用于 v_hourly / v_sizes 视图和聚合兜底。"
                "示例：SELECT * FROM v_hourly WHERE hour >= '2026-09-12T00' ORDER BY hour"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "只读 SELECT 语句"},
                    "limit": {"type": "integer", "description": "行数上限，默认 200"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facts",
            "description": (
                "查询口径定义与数据陷阱（facts.yaml）：正向/反向/成双的定义、"
                "lr_flag 特殊值、shoe_size NULL 的含义等。查数前务必确认口径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "按表名过滤，如 'pieces'"},
                },
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "read_document": read_document,
    "list_tables": list_tables,
    "query_data": query_data,
    "run_sql": run_sql,
    "get_facts": get_facts,
}
