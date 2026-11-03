# 阶段05 性能与失败摘要

- generated_at: 2026-11-03T09:08:05.311641+00:00
- 重跑: `python -B scripts/05_feedback_ab.py`（或 `python -B scripts/run_all.py`）
- 证据: `stage05_feedback_ab.json`、`stage05_feedback_table.md`

## 性能

| 指标 | 值 |
| --- | --- |
| 用例数 | 5 |
| 通过 / 失败 | 5 / 0 |
| p50 / p95 / max 延迟(ms) | 0.562 / 0.628 / 0.628 |
| 计时样本数 | 5 |

- 计时口径：单次响应 elapsed_ms，含会话合并、召回排序与逐条解释。

## 失败与降级

| 类型 | 数量/说明 | 证据 |
| --- | --- | --- |
| 预期错误 | 1：非法反馈 `INVALID_FEEDBACK`，按预期通过 | `expected_error_count=1` |
| 意外失败 | 0 | `unexpected_error_count` |
