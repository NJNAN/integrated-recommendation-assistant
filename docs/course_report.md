# 多模态个性化推荐助手课程报告基础稿

本稿对应学校模板章节，记录模块三的实现与可重跑证据。个人封面、成员贡献、模块一/二原报告、
个人总结和教师评价需按实际情况补充；本稿不能直接替代每位成员的正式 PDF 报告。

## 1 实训项目概述

本项目面向礼物挑选、通勤用品和学习用品推荐场景，提供文字、ASR 转写、视觉标签和确认手势的统一交互。
用户可以逐步给出预算、颜色和对象，查看商品事实证据，并用正负反馈调整当前会话排序。
工程目标是稳定接口、可复现数据流、明确失败出口和完整验收证据。

模块一提供推荐服务，模块二提供多模态识别，模块三通过适配器编排事件与推荐请求。
代码按五个人的模块组织（契约与接入 / 输入与意图 / 召回与编排 / 解释与呈现 / 反馈与服务），
每人一个阶段、一组文件、一组证据，见 `docs/五人分工.md`。

## 2 开发环境与工程规范

离线核心（`IRA_PROFILE=demo` 或未发现同级模块时）只用 Python 3.10+ 标准库、HTML/CSS/JS：
`http.server` 提供服务，`dataclasses` 定义契约，`unittest` 做回归，`urllib` 调上游。
默认 auto 模式会接入同级模块一的冻结 Flask 服务（需运行 `app.py` 的解释器安装 Flask）
和模块二自己的 `.venv` 模型环境，这两部分不属于“零依赖”。

关键配置统一在 `config/settings.py`；脚本可用 `-B` 从任意目录执行；结构化证据写在 `outputs/`；
运行命令、接口协议与降级矩阵见 `README.md` 与 `docs/deployment.md`。

## 3 项目一 推荐系统实践

本轮接入模块一现有冻结 v2 Web 服务，未重跑训练或竞赛指标。适配器通过 `/api/auth/login` 登录、
`/api/recommendations` 分页取候选（每次请求独立 CookieJar，检查登录身份与 `user_id` 一致），
不导入发布包源码、不读模型变量。三个演示用户的商品排序与发布排序一致。

已导入发布目录 2,208 件商品，并从 `development/data/train.csv` 提取同 ID 最后观测的正向交易单价。
价格单位 GBP、带日期与来源，不代表当前售价；缺失价格保持空值并在预算筛选时排除；
颜色只取英文描述里的明确颜色词，未标注的不猜测。人民币预算会先澄清币种，不自动换算。

## 4 项目二 智能多模态 AI 助手

通过独立 JSON-lines 子进程复用模块二现有环境：ASREngine 做真实 SenseVoiceSmall 识别，
VLModel 调本地视觉模型提取品类与颜色，HumanDetector + GestureRecognizer 做稳定手势确认。
低于 0.7 的视觉标签被忽略，评分是模型自报、未经标定；`thumbs_up/down` 映射为推荐反馈。

网页提供录音/WAV 上传、图片上传/拍照、连续手势，只在点击采集按钮时申请权限。
验收脚本使用的是“已转写文本 / 已识别标签 / 已确认手势”的结构化输入，
**不代表物理麦克风、摄像头与真人手势已现场测试**，也不代表模型准确率。

## 5 项目三 多模态个性化推荐助手

### 5.1 场景与边界

主链路：语音转写提出需求 → 融合预算与对象 → 多路召回 → 硬约束过滤 → 排序 → 证据解释 → 点赞反馈重排。
默认接入同级模块一的真实目录与冻结排序；`IRA_PROFILE=demo` 使用 18 件合成商品。业务代码不固定推荐商品。

### 5.2 架构与契约

`InteractionEvent`、`RecommendationRequest`、`RecommendationItem`、`EvidencePack` 定义跨层契约，
`trace_id` 贯穿输入、意图、请求、候选、解释与错误。事件要求 `user_id/session_id/带时区 timestamp/source`，
非法输入返回明确错误码。上下文图、容器图与时序图见 `docs/architecture.md`。

### 5.3 意图与上下文

支持 discover、similar、filter、compare、feedback、explain 六类意图。通过文本规则、显式 slots 和
可信视觉标签融合 category、budget_min/max、recipient、color、scene，并保存来源与置信度。
缺少商品目标追问；预算无法解析追问；同轮颜色/品类冲突或预算上下界冲突拒绝更新。
状态按 `(user_id, session_id)` 隔离，撤销恢复约束，结束清理偏好，默认 TTL 1800 秒。

### 5.4 召回与精排

热门、品类、对象/场景标签、相似属性和模块一排名组成候选来源，按商品 ID 去重。
预算、品类、颜色和缺失价格**先硬过滤**，再计算热门权重、场景匹配、相似度、上游分数和反馈分数，
最终按 `(-score, item_id)` 稳定排序。无候选时提示调整条件，不允许高分绕过预算。

### 5.5 证据解释

解释器只读 `RecommendationItem` 与 `EvidencePack`，用目录属性、描述片段和匹配约束组织文字。
价格与属性必须和证据一致；缺片段或字段不一致时停止解释并返回错误码，不编造。
`speech_text` 与文字一致，供后续 TTS 适配。当前是目录事实检索 + 模板生成，不宣称使用大模型生成。

### 5.6 反馈闭环

点赞/倒赞覆盖当前商品的正负标签，分值 +2.5/-2.5；同方向重复反馈不累积，偏好还会对同标签商品
产生较小影响；只允许反馈当前会话最近展示过的商品。阶段05 固定用户、查询、候选池和 TopK，
记录正反馈提升、重复反馈无变化、反向反馈相对基线下降、非法反馈回滚、跨会话恢复基线，
见 `outputs/stage05_feedback_ab.json`（`rank_change`/`comparison_scope`）与 `stage05_performance_summary.md`。

### 5.7 验收结果

| 阶段 | 通过/样例 | 证据 |
| --- | --- | --- |
| 接口契约 | 6/6 | `stage01_contract_report.json` |
| 意图与上下文 | 30/30 | `stage02_context_cases.jsonl`、`stage02_context_metrics.json`、`stage02_error_cases.md` |
| 编排与召回 | 4/4 | `stage03_demo.json` |
| 证据与解释 | 5/5 | `stage04_explanation_cases.jsonl`、`stage04_abnormal_explanations.md` |
| 反馈对照 | 5/5 | `stage05_feedback_ab.json`、`stage05_performance_summary.md` |
| 端到端 | 7/7 | `stage06_acceptance.json` |

数字对应 `outputs/acceptance_summary.json`，共 **57 条**固定样例（6+30+4+5+5+7）。
阶段02 通过率 100%，达到任务书 ≥80% 的门槛；验收包含期望失败、空集、追问与降级，
这些行为符合预期时计为通过。不能把 100% 解释为真实推荐准确率或模型效果；
延迟只统计有实际响应计时的用例，p50/p95/max 与样本数以 JSON 为准，不含模型与真实网络耗时。
单元/契约/本机 HTTP 回归测试数量以 `outputs/test_results.json` 的 `tests_run` 为准。
各阶段设计、结果、失败案例与证据路径见 `docs/stage_summaries.md`。

### 5.8 失败案例与修复

- 阶段02：无效商品携带新颜色时，旧实现先合并条件再校验目标，失败请求会污染会话；
  改为会话快照 + 错误回滚后，失败后约束保持原值。预算“三百元”未识别金额却继续推荐，
  改为明确追问。同轮颜色/品类冲突与预算上下界冲突改为拒绝更新（`stage02_error_cases.md`）。
- 阶段01：枚举字段传数组时原先触发类型异常，修复为契约错误 `INVALID_EVENT`。
- 阶段03：上游非数值分数原先误记为 `INVALID_EVENT`，修复为 `INVALID_UPSTREAM`；
  上游超时降级为离线召回但保留硬约束（`stage03_demo.json`）。
- 阶段04：无证据、字段缺失或价格与证据不一致时停止解释并降级，不编造属性
  （`stage04_abnormal_explanations.md`）。
- 阶段05：重复反馈不改变排名、反向反馈相对基线下降、非法反馈不污染会话、跨会话恢复基线。
- 阶段06：故障降级、非法输入、会话结束、边界空集用例均为预期行为并计为通过。

## 6 系统交付与工程分析

`README.md` 包含安装/运行、接口、降级与限制；`docs/deployment.md` 包含真实接入、自动回退、
降级矩阵、六维度风险与演示提纲；`docs/stage_summaries.md` 汇总六阶段设计/结果/失败案例/证据路径；
`docs/submission_checklist.md` 列出人工提交事项。默认 auto 模式在模块一启动失败时打印 stderr 告警并回退
`data/catalog.json`；没有同级模块一或显式 `IRA_PROFILE=demo` 时正常离线启动、不产生回退告警。
本地服务无账号认证、只绑回环地址，不适合公网部署；真实输入不持久化，网页导出的 JSON 可能包含输入内容，
应由导出者审查。

后续需要物理麦克风/摄像头与真人手势现场测试、更丰富的独立语言测试集和认证层；
相邻小组复现记录、演示视频和个人 PDF 仍需真实执行。

## 7 个人完成部分

各成员分别填写任务、核心实现、测试证据及至少两个实际问题解决记录；把本稿第5章作为共享技术材料，
按本人真实职责筛选补充，不得把 AI 生成的工程记录直接归属为某位成员的亲历过程。
本稿是**共享基础稿，不能替代正式报告**：每位成员仍需按
`docs/《行业工程实践》普本学生实训报告模板.docx` 独立成稿、补齐封面/表1-3/第7章/第8章并导出 PDF
（命名“小组号-成员名-实训报告.pdf”），流程见 `docs/submission_checklist.md`。
当前目录无 Git 仓库，表7-2 的“Git 提交”行需改用实验记录或结构化输出并在报告中说明。

## 8 实训总结与学习体会

个人总结需结合真实六周任务、团队沟通、失败调试和责任分工撰写（模板建议不少于 800 字），
本稿不替成员虚构学习经历。

## 9 参考资料

1. docs 阶段01—阶段06 课程任务书（需求与验收依据）。
2. docs《行业工程实践》普本学生实训报告模板及材料提交规范。
3. 模块一 `release/web_system` 的公开 HTTP 接口定义（登录、推荐分页）。
4. 模块二 `src/asr.py`、`assistant_service.py`、`gesture_actions.py` 的公开接口与动作定义。
5. 本项目 `docs/architecture.md`、`docs/deployment.md`、源代码及 `outputs/` 结构化证据。

## 附录 证据索引

协议见 `assistant/contracts.py`；会话见 `assistant/session.py`；召回与过滤见 `assistant/recommendation.py`；
解释见 `assistant/explanation.py`；反馈见 `assistant/feedback.py`；HTTP 见 `assistant/server.py`。
六阶段数字对应各 stage 文件与 `acceptance_summary.json`；错误/异常/性能汇总见
`stage02_error_cases.md`、`stage04_abnormal_explanations.md`、`stage05_performance_summary.md`；
反馈排序对应 `stage05_feedback_ab.json` 的 `rank_change`；测试结果见 `test_results.json`。
正式报告还需按模板补齐表A-1 指标追溯索引（表3-2/4-2/5-2/5-3 → outputs 字段）。
