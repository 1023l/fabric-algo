# pieces 表（过线明细）

每个布片穿过计数线时落一条记录。数据来源：检测服务 `/api/algo/count_frame` 的过线事件。

## 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 记录主键 |
| session_id | TEXT | 检测会话 ID（一次 Start 检测 = 一个会话；会话删除时数据同步删除） |
| ts | TEXT | 过线时间，本地时间 ISO 格式（如 `2026-09-12T10:23:45`） |
| track_id | INTEGER | 跟踪 ID（会话内布片编号，跨会话会重复） |
| direction | TEXT | `down`=正向过线（下线产量+1）；`up`=反向过线（布片回退，累计-1） |
| conf | REAL | 该帧检测置信度 |
| text | TEXT | OCR 原文（如 `30-30TR L`；未识别为 NULL） |
| shoe_size | INTEGER | 鞋码 = text 中 `-` 前的数字（`30-30TR` → 30）；货号类文字无鞋码为 NULL |
| lr_flag | TEXT | OCR 文字中含 `L`/`R` 的标记：`L`、`R` 或 `L/R`；无则为 NULL |

## 视图

- `v_hourly` — 按小时聚合：hour, down_cnt, up_cnt, net_cnt, l_cnt, r_cnt, size_kinds
- `v_sizes` — 鞋码分布：shoe_size, down_cnt, up_cnt, total

## 典型查询

```sql
-- 今天每小时产量趋势
SELECT * FROM v_hourly WHERE hour >= '2026-09-12T00' ORDER BY hour;

-- 今天鞋码分布
SELECT * FROM v_sizes ORDER BY total DESC;

-- 某时段正向明细
SELECT ts, track_id, shoe_size, lr_flag, text FROM pieces
WHERE direction='down' AND ts >= '2026-09-12T08:00:00' AND ts <= '2026-09-12T12:00:00';
```
