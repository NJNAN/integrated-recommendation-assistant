# 六阶段验收清单对照表

> 逐条对照 docs 六份任务书的「验收清单」，给出实现位置、可重跑命令和落盘证据。
> 数字一律以 `outputs/` 最新文件为准（重跑 `python -B scripts/run_all.py` 后以 outputs 覆盖本文的示例数字）。

| 阶段 | 验收清单（任务书原文） | 实现位置 | 证据 / 复跑 |
| --- | --- | --- | --- |
| 01 | 两套系统不直接读取对方内部变量 | `assistant/adapters.py`（HTTP 公开接口）、`assistant/module_two.py`（JSON-lines 子进程） | 代码审查；`GET /api/health` 显示上游状态 |
| 01 | 字段包含 trace_id、user_id、timestamp、source | `assistant/contracts.py::InteractionEvent` | `outputs/stage01_contract_report.json` 缺字段用例 |
| 01 | 非法输入返回明确错误 | `contracts.py` 校验 + `orchestrator.py` 错误分支 | `stage01_contract_report.json` 中 `INVALID_EVENT` 用例 |
| 01 | 离线演示模式可运行 | `config/settings.py::offline_demo_settings`、`app.py::prepare` | `$env:IRA_PROFILE='demo'; python -B app.py` |
| 02 | 必选槽位缺失时不臆测 | `understanding.py::missing`、`orchestrator.py` 追问分支 | `outputs/stage02_context_cases.jsonl`（预算“三百元”、缺 item_id） |
| 02 | thumbs_up/down 映射为正负反馈 | `understanding.py`（手势→feedback）、`feedback.py::apply_feedback` | `stage02_context_cases.jsonl` 点赞/倒赞用例 |
| 02 | 约束冲突可检测 | `session.py::merge`、`understanding.py::conflicts` | `stage02_context_cases.jsonl` 视觉冲突/预算冲突 |
| 02 | 20 条样例通过率不低于 80% | `scripts/02_context_test.py`（≥20 条固定样例） | `outputs/stage02_context_metrics.json`（`case_count/pass_rate`） |
| 03 | 候选来源可追溯 | `recommendation.py` 的 `sources` + `candidates` 诊断 | `outputs/stage03_demo.json` 每件商品 `sources` |
| 03 | 硬约束不被排序分数覆盖 | `recommendation.py` 先过滤后打分 | `stage03_demo.json` 空集用例、`tests/test_recommendation.py` |
| 03 | 空结果有清晰提示 | `orchestrator.py` 空结果文案（含“调整”） | `stage03_demo.json`、`stage06_acceptance.json` |
| 03 | 同输入和配置可复现 | 排序 `(-score, item_id)`、无随机 | `stage03_demo.json` 的 `sha256.first==second` |
| 04 | 每条解释至少有一项可核验证据 | `explanation.py` 引用必须属于本商品 | `outputs/stage04_evidence_table.md` |
| 04 | 无证据时不编造 | `explanation.py::INSUFFICIENT_EVIDENCE` | `stage04_abnormal_explanations.md` |
| 04 | 商品价格与属性一致 | `explanation.py` 属性比对 → `EVIDENCE_MISMATCH` | `stage04_explanation_cases.jsonl` 价格+1 用例 |
| 04 | 视觉标签低置信度时降权或忽略 | `settings.visual_threshold=0.7` + `recommendation.py` 过滤 | `stage04_explanation_cases.jsonl` 弱视觉用例 |
| 05 | 反馈后排序变化可解释 | `recommendation.py` 的 `feedback`/`tag_affinity` 分数明细 | `outputs/stage05_feedback_ab.json`、`stage05_feedback_table.md` |
| 05 | 负反馈不会永久污染其他会话 | `session.py` 按 `(user_id, session_id)` 隔离 | `stage05_feedback_ab.json` 会话隔离用例 |
| 05 | 对照口径一致 | 脚本固定用户/查询/候选池/TopK | `stage05_feedback_ab.json` 的 `comparison_scope` |
| 05 | 至少包含一个无变化或失败案例 | 重复反馈无变化、非法反馈 `INVALID_FEEDBACK` | `stage05_feedback_ab.json` 5 条用例 |
| 06 | 三类输入链路可运行 | `media.py` + `module_two.py` + 前端四种输入 | `outputs/stage06_acceptance.json` voice/vision/gesture |
| 06 | 至少一种故障可降级 | `TimeoutRemote` 注入 → 离线召回 | `stage06_acceptance.json` 故障降级用例 |
| 06 | 报告数字可追溯 | 所有数字来自 `outputs/` | `outputs/acceptance_summary.json`、`test_results.json` |
| 06 | 无缓存、模型和敏感数据进入提交包 | `.gitignore` + [提交清单](submission_checklist.md) | 打包前手工检查 |

## 各阶段提交物

| 阶段 | 任务书要求的提交物 | 本仓库位置 |
| --- | --- | --- |
| 01 | `docs/architecture.md`、`outputs/stage01_contract_report.json`、契约测试日志 | 同名文件 + `outputs/stage01_contract_test.log` |
| 02 | `outputs/stage02_context_cases.jsonl`、`stage02_context_metrics.json`、错误案例分析 | 同名 + `outputs/stage02_error_cases.md` |
| 03 | `outputs/stage03_demo.json`、三类链路 trace、超时与降级说明 | 同名 + `docs/deployment.md` 降级矩阵 |
| 04 | 解释与证据对照表、`outputs/stage04_explanation_cases.jsonl`、异常解释记录 | `outputs/stage04_evidence_table.md`、同名、`stage04_abnormal_explanations.md` |
| 05 | `outputs/stage05_feedback_ab.json`、反馈前后对照表、性能与失败案例摘要 | 同名 + `stage05_feedback_table.md`、`stage05_performance_summary.md` |
| 06 | `outputs/stage06_acceptance.json`、部署与风险文档、课程总项目报告、演示材料 | 同名 + `docs/deployment.md`、`docs/course_report.md`、`docs/deployment.md` 第5节 |

## 复跑命令

```powershell
python -B scripts/run_all.py            # 六阶段全部重跑，任一失败非零退出
python -B scripts/01_contract_test.py   # 也可单独重跑某个阶段
python -B -m unittest discover -s tests -v
python -B scripts/test_report.py        # 回归测试结果落盘
```

## 已知边界（报告里要写清楚）

- 阶段样例使用 18 件合成商品与固定规则，只证明契约与流程正确，**不代表真实推荐效果**。
- 真实目录模式的价格是 GBP 历史最后观测单价，不是当前售价；人民币预算会先澄清币种，不自动换算。
- 语音/视觉/手势的验收输入是“已转写文本 / 已识别标签 / 已确认手势”，真实麦克风、摄像头与真人手势
  需现场演示时另行确认（`stage06_acceptance.json` 的 `scope.real_models_verified=false`）。
