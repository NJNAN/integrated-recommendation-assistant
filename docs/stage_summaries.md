# 六阶段阶段小结（设计、结果、失败案例与证据路径）

> 对应六份任务书“步骤5：阶段收尾”的第 3 项：在总报告中补充本阶段设计、结果、失败案例与证据路径。
> 数字一律以 `outputs/` 最新文件为准（重跑 `python -B scripts/run_all.py` 后以 outputs 覆盖本文示例）。

当前：**57 条固定样例全部通过**（阶段 01–06 分别为 6 / 30 / 4 / 5 / 5 / 7，其中阶段02 要求 ≥20 条、
通过率 ≥80%），汇总见 `outputs/acceptance_summary.json`；回归测试数量以 `outputs/test_results.json` 为准。

| 阶段 | 设计要点 | 结果 | 失败 / 边界案例 | 证据路径 |
| --- | --- | --- | --- | --- |
| 01 需求分析、架构与接口契约 | 统一 `InteractionEvent/RecommendationRequest/RecommendationItem/EvidencePack`；`trace_id/user_id/session_id/timestamp/source` 必填；错误码；适配器边界；离线降级 | 6/6 | 缺 `trace_id`/`source`、视觉置信度 1.1 → `INVALID_EVENT`；预算 0 → `empty` 且约束保留；`intent=filter` 无槽位 → 追问 | `outputs/stage01_contract_report.json`、`stage01_contract_test.log`、`docs/architecture.md` |
| 02 多模态意图与用户上下文 | 六类意图 + 品类/预算/对象/颜色/场景槽位与来源置信度；`SessionContext` 多轮合并、撤销、冲突、结束清理；手势映射正负反馈 | 30/30 | 预算“三百元”无法解析→追问；同轮颜色冲突→`conflict`；多轮预算冲突不污染旧约束；低置信标签进 `ignored_labels` | `outputs/stage02_context_cases.jsonl`、`stage02_context_metrics.json`、`stage02_error_cases.md` |
| 03 对话编排、多路召回与可控精排 | 主链路编排；popularity/品类/场景/相似/上游多路召回；去重与 TopK；硬约束先过滤再打分；无结果与上游超时降级；确定性复现 | 4/4 | 硬约束空集不放宽预算；上游超时降级为 popularity 召回且保留约束；同输入两次响应 sha256 一致 | `outputs/stage03_demo.json`、`docs/deployment.md` 降级矩阵 |
| 04 商品多模态理解、RAG 解释与交互呈现 | `EvidencePack`（属性/来源/匹配约束/视觉标签/片段）；解释器只读 `RecommendationItem + EvidencePack`；证据不足降级；文本与语音同文本 | 5/5 | 清空证据→`INSUFFICIENT_EVIDENCE`；删价格/价格+1→`EVIDENCE_MISMATCH`；低置信视觉标签不进入证据 | `outputs/stage04_explanation_cases.jsonl`、`stage04_evidence_table.md`、`stage04_abnormal_explanations.md` |
| 05 反馈闭环、效果评估与工程优化 | 正负反馈写入当前会话（覆盖不累加）；固定用户/查询/候选池/TopK 做反馈前后对照；无变化与非法反馈案例；性能摘要 | 5/5 | 重复同方向反馈分数不变；反向反馈排名低于基线；非法反馈 `INVALID_FEEDBACK` 且状态回滚；新会话恢复基线 | `outputs/stage05_feedback_ab.json`、`stage05_feedback_table.md`、`stage05_performance_summary.md` |
| 06 系统联调、验收部署与课程总答辩 | 端到端 voice/vision/gesture 三链路 + 故障降级 + 非法输入 + 会话结束 + 边界空集；`trace_id` 全链路断言；部署/风险/演示材料 | 7/7 | 上游故障→降级且 trace 保留；非法输入→`error`；空集→`empty` 且约束保持 | `outputs/stage06_acceptance.json`、`acceptance_summary.json`、`docs/deployment.md` |

## 阶段间复用关系

- 01→02：统一事件、错误码与离线降级被阶段02 直接复用，不新增协议。
- 02→03：`SessionContext` 的硬约束/软偏好/撤销结果作为 `RecommendationRequest` 输入，编排只消费结构化状态。
- 03→04：`RecommendationItem` 的 `sources/score/score_breakdown/evidence` 是解释器的唯一事实来源。
- 04→05：解释引用的商品ID与证据用于反馈对照，对照固定同一用户/查询/候选池/TopK。
- 05→06：反馈对照与性能摘要进入端到端验收，`trace` 七阶段
  （event→intent→context→request→recall_and_ranking→explanation→result）串联全链路。
- 06→交付：outputs 证据、阶段小结与提交清单进入总报告附录与提交包。

## 使用说明

1. 总报告第 5 章或附录可逐阶段引用本表四列（设计/结果/失败案例/证据路径），再按模板改写为报告语体。
2. 引用数字时写“见 outputs/xxx.json 字段”，不要复制过期数字；重跑 `run_all.py` 后以 outputs 为准。
3. 阶段 02/04/05 的独立提交物已落盘为 `stage02_error_cases.md`、`stage04_abnormal_explanations.md`、
   `stage05_performance_summary.md`，与 [提交清单](submission_checklist.md) 一并提交。
