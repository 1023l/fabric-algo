"""产线数据问答 Agent 包。

数据流：count_frame 过线事件 → store.py 落 SQLite → tools.py 查询 →
agent runtime(ReAct) → qa_routes.py SSE 输出到对话页。

TODO(eval)：黄金测试集 + 评测脚本（参照 bi-cli eval/run_golden.py 的流程）暂缓——
等上了产线、有真实数据后再照搬流程补齐；当前只有框架骨架，属刻意为之。
"""

from .store import (  # noqa: F401
    record_piece, update_piece_text, touch_session, remove_session,
    query_pieces, stats_summary,
)
