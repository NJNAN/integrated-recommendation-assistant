# 阶段02 错误案例分析

- 重跑: `python -B scripts/02_context_test.py`（或 `python -B scripts/run_all.py`）
- 证据: `outputs/stage02_context_cases.jsonl`、`outputs/stage02_context_metrics.json`
- 规模: case_count=30，passed=30，failed=0，pass_rate=1.0

## 错误、追问与回滚案例

| 分类 | 用例 | 期望状态 | 实际状态 | 消息/错误码 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 追问 | 弱视觉 | needs_clarification | needs_clarification | 请补充：可信视觉标签或补充文字 | PASS |
| 冲突 | 视觉冲突 | conflict | conflict | 同一轮输入包含矛盾条件，请确认品类、颜色或预算。 | PASS |
| 追问 | 缺少反馈对象 | needs_clarification | needs_clarification | 请补充：item_id | PASS |
| 追问 | 缺少相似对象 | needs_clarification | needs_clarification | 请补充：item_id | PASS |
| 追问 | 缺少比较对象 | needs_clarification | needs_clarification | 请补充：至少两个 item_ids | PASS |
| 冲突 | 预算冲突 | conflict | conflict | 预算下限高于上限，请修改预算或撤销上一步 | PASS |
| 空集 | 低预算空集 | empty | empty | 没有符合当前品类、预算和颜色的商品。请调整条件或撤销上一步；硬约束保持有效。 | PASS |
| 追问 | 预算未解析 | needs_clarification | needs_clarification | 请补充：预算金额（请使用阿拉伯数字，例如300元以内） | PASS |
| 冲突 | 多轮冲突不污染 | conflict | conflict | 预算下限高于上限，请修改预算或撤销上一步 | PASS |
| 回滚/清理 | 结束清理后新事件恢复基线 | ended | ended | 会话已清理，可以重新开始。 | PASS |

## 失败出口说明

- 冲突：同轮预算上下界矛盾或视觉标签与文字颜色矛盾返回 `conflict`，不写入约束；多轮冲突保持上一轮有效约束，失败请求不污染会话。
- 追问：低置信标签被忽略、缺少反馈/相似/比较对象、预算无法解析时返回 `needs_clarification`，只报告缺失字段，不猜测用户意图。
- 空集：低预算等硬约束无商品时返回 `empty`，提示“调整条件”，不擅自放宽约束。
- 回滚/清理：冲突后约束不变；结束会话后新事件重建空约束、空偏好的会话（见 JSONL 的 followup_response）。

## 复核方法

```powershell
python -B scripts/02_context_test.py
```
