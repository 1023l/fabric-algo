"""检测计数结果 SQLite 落库（问答 Agent 的数据源）。

设计：
- 只在过线事件（crossed=True）时写一行，频率低（每片一次），不拖累检测主流程
- 线程安全：web_server/algo_routes 在多线程里调用，用模块级锁串行化
- 主流程保护：调用方（algo_routes）应 try/except 包住，DB 异常不影响检测
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根（fabric-algo/）

DB_PATH = ROOT / "qa_agent" / "data" / "qa.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pieces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,              -- ISO 本地时间，如 2026-09-12T10:23:45
    track_id INTEGER NOT NULL,
    direction TEXT NOT NULL,       -- down(正向) / up(反向)
    conf REAL,
    text TEXT,                     -- OCR 原文（可为 NULL）
    shoe_size INTEGER,             -- 提取的鞋码（可为 NULL）
    lr_flag TEXT                   -- L / R / L/R（可为 NULL）
);
CREATE INDEX IF NOT EXISTS idx_pieces_ts ON pieces(ts);
CREATE INDEX IF NOT EXISTS idx_pieces_session ON pieces(session_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT,
    line_ratio REAL
);

-- 时段聚合视图（问答常用）
CREATE VIEW IF NOT EXISTS v_hourly AS
SELECT substr(ts, 1, 13) || ':00' AS hour,
       SUM(direction = 'down') AS down_cnt,
       SUM(direction = 'up')   AS up_cnt,
       SUM(direction = 'down') - SUM(direction = 'up') AS net_cnt,
       SUM(lr_flag LIKE '%L%') AS l_cnt,
       SUM(lr_flag LIKE '%R%') AS r_cnt,
       COUNT(DISTINCT shoe_size) AS size_kinds
FROM pieces
GROUP BY substr(ts, 1, 13);

-- 鞋码分布视图
CREATE VIEW IF NOT EXISTS v_sizes AS
SELECT shoe_size,
       SUM(direction = 'down') AS down_cnt,
       SUM(direction = 'up')   AS up_cnt,
       COUNT(*) AS total
FROM pieces
WHERE shoe_size IS NOT NULL
GROUP BY shoe_size;
"""


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 写入（algo_routes 过线事件调用）
# ---------------------------------------------------------------------------

def touch_session(session_id: str, line_ratio: float | None = None) -> None:
    """登记/更新会话（count_frame 每帧调用，UPSERT）。"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sessions(session_id, created_at, last_active_at, line_ratio) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at, "
            "line_ratio=COALESCE(excluded.line_ratio, line_ratio)",
            (session_id, _now(), _now(), line_ratio),
        )
        conn.commit()


def record_piece(session_id: str, track_id: int, direction: str, conf: float | None,
                 text: str | None, shoe_size: int | None, lr_flag: str | None) -> None:
    """过线事件落库（每个新过线布片一行）。"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO pieces(session_id, ts, track_id, direction, conf, text, shoe_size, lr_flag) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (session_id, _now(), int(track_id), direction, conf, text, shoe_size, lr_flag),
        )
        conn.commit()


def update_piece_text(session_id: str, track_id: int, text: str | None,
                      shoe_size: int | None, lr_flag: str | None) -> bool:
    """OCR 异步识别成功后回填最近一条该 track 的 text 为空的记录。

    过线当帧 OCR 还没出结果（pending_ocr 窗口内后续帧才识别成功），
    所以先落 direction，识别成功后用本函数补齐 text/鞋码/左右脚。
    """
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE pieces SET text=?, shoe_size=?, lr_flag=? WHERE id=("
            "SELECT id FROM pieces WHERE session_id=? AND track_id=? AND text IS NULL "
            "ORDER BY id DESC LIMIT 1)",
            (text, shoe_size, lr_flag, session_id, int(track_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def remove_session(session_id: str) -> int:
    """删除会话及其全部数据（对齐 DELETE /api/algo/count_sessions/{sid}）。"""
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM pieces WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# 查询（agent tools 调用）
# ---------------------------------------------------------------------------

# 白名单：列名 -> SQL 片段（防注入，只允许这些字段出现在 SELECT/WHERE/ORDER BY）
_PIECE_COLS = {
    "id": "id", "session_id": "session_id", "ts": "ts", "track_id": "track_id",
    "direction": "direction", "conf": "conf", "text": "text",
    "shoe_size": "shoe_size", "lr_flag": "lr_flag",
}


def query_pieces(*, fields: list[str] | None = None,
                 ts_start: str | None = None, ts_end: str | None = None,
                 direction: str | None = None, shoe_size: int | None = None,
                 session_id: str | None = None,
                 group_by: list[str] | None = None,
                 order_by: str | None = None,
                 limit: int = 200) -> dict:
    """结构化查询 pieces 表（全部参数化，字段走白名单）。"""
    cols = fields or ["ts", "direction", "shoe_size", "lr_flag", "text"]
    bad = [c for c in cols if c not in _PIECE_COLS]
    if bad:
        return {"error": f"未知字段: {bad}", "allowed": list(_PIECE_COLS)}

    where, params = [], []
    if ts_start:
        where.append("ts >= ?"); params.append(ts_start)
    if ts_end:
        where.append("ts <= ?"); params.append(ts_end)
    if direction in ("down", "up"):
        where.append("direction = ?"); params.append(direction)
    if shoe_size is not None:
        where.append("shoe_size = ?"); params.append(int(shoe_size))
    if session_id:
        where.append("session_id = ?"); params.append(session_id)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    gb = [g for g in (group_by or []) if g in _PIECE_COLS]
    if gb:
        select_sql = ", ".join(f"{_PIECE_COLS[c]} AS {c}" for c in cols) + \
                     f", COUNT(*) AS cnt"
        group_sql = f" GROUP BY {', '.join(_PIECE_COLS[c] for c in gb)}"
    else:
        select_sql = ", ".join(f"{_PIECE_COLS[c]} AS {c}" for c in cols)
        group_sql = ""

    order_sql = ""
    if order_by:
        desc = order_by.startswith("-")
        col = order_by.lstrip("-")
        if col in _PIECE_COLS:
            order_sql = f" ORDER BY {_PIECE_COLS[col]}{' DESC' if desc else ''}"

    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            f"SELECT {select_sql} FROM pieces {where_sql}{group_sql}{order_sql} LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
    return {"count": len(rows), "rows": [dict(r) for r in rows]}


def stats_summary(*, ts_start: str | None = None, ts_end: str | None = None,
                  session_id: str | None = None) -> dict:
    """汇总统计：正向/反向/累计/L/R/成双/单只/鞋码分布（等价界面 summary 的落库版）。"""
    where, params = [], []
    if ts_start:
        where.append("ts >= ?"); params.append(ts_start)
    if ts_end:
        where.append("ts <= ?"); params.append(ts_end)
    if session_id:
        where.append("session_id = ?"); params.append(session_id)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with _lock:
        conn = _get_conn()
        row = conn.execute(
            f"SELECT SUM(direction='down') AS down, SUM(direction='up') AS up, "
            f"SUM(lr_flag LIKE '%L%') AS l_cnt, SUM(lr_flag LIKE '%R%') AS r_cnt "
            f"FROM pieces {where_sql}", params,
        ).fetchone()
        if where_sql:
            size_sql = f"SELECT shoe_size, COUNT(*) AS cnt FROM pieces {where_sql} AND shoe_size IS NOT NULL"
        else:
            size_sql = "SELECT shoe_size, COUNT(*) AS cnt FROM pieces WHERE shoe_size IS NOT NULL"
        sizes = conn.execute(size_sql, params).fetchall()

    down, up = int(row["down"] or 0), int(row["up"] or 0)
    l_cnt, r_cnt = int(row["l_cnt"] or 0), int(row["r_cnt"] or 0)
    return {
        "down": down, "up": up, "net": down - up,
        "lr_l": l_cnt, "lr_r": r_cnt,
        "pairs": min(l_cnt, r_cnt), "single": abs(l_cnt - r_cnt),
        "sizes": {int(r["shoe_size"]): int(r["cnt"]) for r in sizes},
        "ts_start": ts_start, "ts_end": ts_end, "session_id": session_id,
    }


def raw_sql(sql: str, limit: int = 200) -> dict:
    """受限只读 SQL（视图/聚合兜底用）：仅允许 SELECT，禁止写操作。"""
    s = sql.strip().rstrip(";")
    if not s.lower().startswith(("select", "with")):
        return {"error": "只允许 SELECT 查询"}
    forbidden = ("insert", "update", "delete", "drop", "alter", "create",
                 "attach", "pragma", "replace")
    if any(k in s.lower() for k in forbidden):
        return {"error": "包含禁止的关键字"}
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(s + f" LIMIT {int(limit)}").fetchall()
        except sqlite3.Error as e:
            return {"error": f"SQL 错误: {e}"}
    return {"count": len(rows), "rows": [dict(r) for r in rows]}


if __name__ == "__main__":
    # 自检：建表 + 插样例 + 查询
    touch_session("selftest", 0.5)
    record_piece("selftest", 1, "down", 0.9, "30-30TR L", 30, "L")
    record_piece("selftest", 2, "down", 0.9, "30-30TR R", 30, "R")
    record_piece("selftest", 3, "up", 0.9, None, None, None)
    print(json.dumps(stats_summary(session_id="selftest"), ensure_ascii=False, indent=2))
    print(json.dumps(query_pieces(session_id="selftest"), ensure_ascii=False))
    print(json.dumps(raw_sql("SELECT * FROM v_hourly"), ensure_ascii=False))
    print(json.dumps(raw_sql("SELECT * FROM v_sizes"), ensure_ascii=False))
    remove_session("selftest")
