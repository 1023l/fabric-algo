"""产线数据问答 Agent 包。

数据流：count_frame 过线事件 → store.py 落 SQLite → tools.py 查询 →
agent runtime(ReAct) → qa_routes.py SSE 输出到对话页。
"""

from .store import (  # noqa: F401
    record_piece, update_piece_text, touch_session, remove_session,
    query_pieces, stats_summary,
)
