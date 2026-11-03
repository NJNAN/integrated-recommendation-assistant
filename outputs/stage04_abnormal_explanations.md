# 阶段04 异常解释记录

- generated_at: 2026-11-03T01:33:20.174736+00:00
- 重跑命令: `python -B scripts/04_explanation_test.py`
- 证据路径: `outputs/stage04_explanation_cases.jsonl`、`outputs/stage04_evidence_table.md`

| 用例 | 注入 | 期望错误码 | 实际错误码 | 引用数 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 无证据不编造 | evidence.snippets=[] | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | 0 | PASS |
| 商品字段缺失 | evidence.attributes.price=<deleted> | EVIDENCE_MISMATCH | EVIDENCE_MISMATCH | 0 | PASS |
| 价格不一致 | evidence.attributes.price=<original+1> | EVIDENCE_MISMATCH | EVIDENCE_MISMATCH | 0 | PASS |
| 弱视觉商品标签忽略 | 低置信视觉标签（0.7 阈值以下忽略） | OK | ok | 0 | PASS |

## 降级原则

- 缺证据片段或属性时停止解释并返回 `INSUFFICIENT_EVIDENCE`，不编造引用。
- 属性与证据不一致（缺字段、价格整加一）时返回 `EVIDENCE_MISMATCH`，引用清空。
- 视觉标签低于阈值不进入证据，正常解释继续；异常用例均记录注入内容与判定。
