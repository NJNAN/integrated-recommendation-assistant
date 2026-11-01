# 拾物 · 多模态个性化推荐助手

按 docs 六阶段任务书实现的课程整合项目：把「模块一 推荐系统」与「模块二 多模态助手」整合成
一个可离线演示、可追踪、可复现的系统，支持多轮推荐、语音/图片/手势输入、硬约束过滤、
商品证据解释、正负反馈重排和六阶段可重跑证据。

代码按**五个人的模块**组织，每块一个阶段、一组文件、一组证据，答辩时各讲各的、互不重叠：
见 [docs/五人分工.md](docs/五人分工.md)。设计原则是**能讲清楚**：没有多余抽象，标准库优先，
每个文件只做一件事，函数短小，所有失败都带错误码。

- 六阶段符合性逐条对照：[docs/requirements_check.md](docs/requirements_check.md)
- 架构与接口契约（阶段1 提交物）：[docs/architecture.md](docs/architecture.md)
- **代码阅读指南（30 分钟读懂整个仓库）**：[docs/代码阅读指南.md](docs/代码阅读指南.md)
- 部署、降级、风险与演示提纲：[docs/deployment.md](docs/deployment.md)
- 六阶段小结：[docs/stage_summaries.md](docs/stage_summaries.md)
- 提交清单：[docs/submission_checklist.md](docs/submission_checklist.md)
- GitHub 仓库：<https://github.com/NJNAN/integrated-recommendation-assistant>

## 三个项目的代码都在这个仓库里

整合项目要调两个兄弟项目，所以它们的**代码副本**放在 `vendor/` 下（只读，供阅读与对照）：

| 目录 | 是哪个项目 | 里面有什么 | 怎么被用 |
| --- | --- | --- | --- |
| `assistant/` `web/` `scripts/` `tests/` | **模块三**（本仓库主体） | 整合、编排、前端、六阶段证据 | 直接运行 |
| `vendor/01_recommender_system/` | **模块一 推荐系统** | 冻结 Flask 服务（`release/web_system/`）、Oblivionis v2 冻结排序结果、2,208 件商品目录、模型源码 | 模块三通过 HTTP（登录 + 推荐分页）调用 |
| `vendor/02_multimodal_ai_assistant/` | **模块二 多模态助手** | `src/` 40 个源文件（ASR、视觉、手势、RAG…）、`config/` | 模块三通过 JSON-lines 子进程调用 |

`config/settings.py` 的规则：**同级目录优先，没有就退回 `vendor/`**——开发机用原项目，
别人 clone 这个仓库也能看到、甚至跑起来（需要 Flask）。体积原因未纳入：模块一 173 MB 模型 pickle、
72 MB 原始交易数据、模块二 `.venv` 与模型权重；详见 [vendor/README.md](vendor/README.md)。

## 安装与运行

- Python 3.10+（实测 3.13.6）。离线核心、HTTP 服务、测试与六阶段脚本**只用标准库**。
- **默认 auto 模式**：优先用同级 `01_recommender_system`，没有就用仓库里的 `vendor/01_recommender_system`，
  接入 2,208 件真实商品与 Oblivionis v2 冻结排序。模块一冻结 Flask 服务由运行 `app.py` 的同一个解释器启动，
  因此该解释器需要 Flask（任一路径下都一样）。
- **模块二（语音/视觉/手势）**：需要模块二自己的 `.venv`（原项目里，或 `IRA_MODULE_TWO_PYTHON` 指向它）；
  仓库里的 `vendor/02…/src` 只有源码，没有 venv，所以媒体接口会返回 503 并保留手动输入。
- **纯离线模式**：`IRA_PROFILE=demo`，不启动模块一、不加载模型，只用 `data/catalog.json` 的 18 件合成商品。

```powershell
python -m pip install -r ..\01_recommender_system\release\requirements-web.txt   # 仅 auto 模式需要
python -B app.py                       # 打开 http://127.0.0.1:8080

$env:IRA_PROFILE = 'demo'; python -B app.py    # 纯离线演示（只要标准库）

python -B scripts/run_all.py                   # 六阶段证据全部重跑
python -B -m unittest discover -s tests -v     # 回归测试
```

## 五人分工与文件

| 成员 | 负责阶段 | 负责文件 | 答辩讲什么 |
| --- | --- | --- | --- |
| 成员1 契约与接入 | 阶段1 | `assistant/contracts.py`、`config/settings.py`、`assistant/adapters.py`、`app.py`、`scripts/01_contract_test.py` | 四类数据结构、错误码、trace_id、两个外部项目的适配器边界、启动与回退 |
| 成员2 输入与意图 | 阶段2 | `assistant/understanding.py`、`assistant/session.py`、`assistant/media.py`、`assistant/module_two.py`、`scripts/02_context_test.py` | 六类意图与槽位、低置信视觉标签、多轮合并/撤销/冲突、媒体入口与模块二 IPC |
| 成员3 召回与编排 | 阶段3 | `assistant/module_one.py`、`assistant/recommendation.py`、`assistant/orchestrator.py`、`scripts/03_orchestrator_demo.py` | 真实目录导入、多路召回、硬约束先于打分、打分公式、主链路编排与降级 |
| 成员4 解释与呈现 | 阶段4 | `assistant/explanation.py`、`web/app.js`、`web/index.html`、`scripts/04_explanation_test.py` | EvidencePack 如何约束解释、无证据不编造、前端如何调 API 展示证据 |
| 成员5 反馈与服务 | 阶段5+6 | `assistant/feedback.py`、`assistant/server.py`、`scripts/05_feedback_ab.py`、`scripts/06_e2e_acceptance.py`、`scripts/run_all.py` | 反馈覆盖不累加、A/B 对照口径、HTTP 路由与错误映射、端到端验收 |

每个成员另有自己的测试文件（`tests/test_*.py`）。详细行数、文件清单和讲解顺序见
[docs/五人分工.md](docs/五人分工.md)。

## 六阶段交付

| 阶段 | 命令 | 主要输出 |
| --- | --- | --- |
| 01 接口契约 | `python -B scripts/01_contract_test.py` | `outputs/stage01_contract_report.json`、`stage01_contract_test.log` |
| 02 意图与上下文 | `python -B scripts/02_context_test.py` | `outputs/stage02_context_cases.jsonl`、`stage02_context_metrics.json`、`stage02_error_cases.md` |
| 03 编排与召回 | `python -B scripts/03_orchestrator_demo.py` | `outputs/stage03_demo.json` |
| 04 证据与解释 | `python -B scripts/04_explanation_test.py` | `outputs/stage04_explanation_cases.jsonl`、`stage04_evidence_table.md`、`stage04_abnormal_explanations.md` |
| 05 反馈对照 | `python -B scripts/05_feedback_ab.py` | `outputs/stage05_feedback_ab.json`、`stage05_feedback_table.md`、`stage05_performance_summary.md` |
| 06 端到端验收 | `python -B scripts/06_e2e_acceptance.py` | `outputs/stage06_acceptance.json` |

汇总为 `outputs/acceptance_summary.json`（各阶段 `case_count/passed` 与 `total_*`）。
当前 **57 条**固定样例（6+30+4+5+5+7，其中阶段02 要求 ≥20 条、通过率 ≥80%），
回归测试数量以 `outputs/test_results.json` 的 `tests_run` 为准。
所有脚本单独执行、失败以非零状态退出、成功落盘结构化证据。

## API 输入与输出

`POST /api/events`，Content-Type: application/json：

```json
{
  "trace_id": "your-request-id",
  "user_id": "demo-user",
  "session_id": "shopping-1",
  "timestamp": "2026-10-08T10:00:00+08:00",
  "source": "voice",
  "payload": {"transcript": "给朋友买礼物，预算300元以内"}
}
```

返回 `status/message/items/context/trace_id/trace/warnings/degraded/elapsed_ms`。
`source` 为 text/voice/vision/gesture；支持 discover/similar/filter/compare/feedback/explain 六类意图。
缺必选槽位会追问（不猜），预算上下界冲突拒绝更新，品类/颜色/预算是硬约束，正负偏好只影响当前会话。

- 视觉：`{"labels":[{"label":"背包","confidence":0.95}]}`
- 手势：`{"gesture":"thumbs_up","item_id":"从推荐结果取到的商品ID"}`
- 显式约束：`{"slots":{"category":"背包","budget_max":300}}`

媒体接口：`POST /api/media/voice|vision|gesture`（base64 音频/图片，走模块二本地识别）；
健康检查：`GET /api/health`（模式、目录大小、模块二环境、演示用户）。

## 降级与失败出口

| 场景 | 行为 | 观察点 |
| --- | --- | --- |
| 未发现模块一 / `IRA_PROFILE=demo` | 用 18 件合成目录正常离线启动 | `/api/health` 的 `mode=offline_demo` |
| 有模块一但启动失败 | stderr 打印回退警告，改用合成目录继续服务 | 启动日志 |
| 推荐上游超时/不可用 | 离线 popularity 召回，`degraded=true` + 错误码 | `outputs/stage03_demo.json` |
| 模块二不可用 | 媒体接口 503，页面保留手动输入 | `/api/health` 的 `module_two.configured` |
| 视觉置信度 < 0.7 | 忽略标签并记录，不进入约束 | `outputs/stage02_context_cases.jsonl` |
| 硬约束无候选 | `status=empty`，不放宽约束 | `outputs/stage03_demo.json` |
| 解释证据不足/属性矛盾 | 停止解释、清空引用、返回错误码 | `outputs/stage04_abnormal_explanations.md` |

完整降级矩阵、风险登记表与演示提纲见 [docs/deployment.md](docs/deployment.md)。

## 目录结构

```text
app.py                 入口：选运行模式 → 起 HTTP → 接入失败回退
config/settings.py     路径、端口、阈值（唯一配置源）+ 兄弟项目/vendor 自动发现
assistant/             契约、意图、会话、召回、编排、解释、反馈、媒体、适配器、HTTP（共 13 个文件）
integrations/          模块一/模块二的独立进程入口（很薄，复用其公开能力）
vendor/01_recommender_system/  模块一代码副本：冻结 Flask 服务 + 冻结排序结果 + 商品目录 + 模型源码
vendor/02_multimodal_ai_assistant/  模块二代码副本：ASR / 视觉 / 手势 / RAG 等 40 个源文件
web/                   前端：index.html + app.js + style.css
scripts/               01~06 六阶段证据脚本 + run_all + test_report
tests/                 五个模块各自的回归测试
outputs/               实测 JSON/JSONL/日志/对照表
docs/                  任务书、架构、部署、五人分工、阶段小结、提交清单
data/                  合成演示目录 / 模块一真实目录（生成）
```

环境变量：`IRA_PROFILE`（auto/demo）、`IRA_CATALOG`、`IRA_OUTPUT_DIR`、`IRA_PORT`、
`IRA_RECOMMENDER_URL`、`IRA_RECOMMENDER_USERNAME`、`IRA_RECOMMENDER_PASSWORD`、`IRA_TIMEOUT`、
`IRA_MODULE_ONE_ROOT`、`IRA_MODULE_ONE_PORT`、`IRA_MODULE_TWO_ROOT`、`IRA_MODULE_TWO_PYTHON`、
`IRA_MODULE_TWO_TEXT_PORT`、`IRA_MODULE_TWO_VISION_PORT`、`IRA_MEDIA_TIMEOUT`。

## 提交

学校要求：`小组号-项目名` 目录、源代码（含 `requirements.txt`、`README.md`）、每人一份 PDF 实训报告；
打包前确认没有 `runtime/`、缓存、`.env`、模型权重和个人音视频。逐项清单见
[docs/submission_checklist.md](docs/submission_checklist.md)。

## 已知边界

- 阶段样例使用 18 件合成商品与固定规则，只证明契约与流程正确，**不代表真实推荐效果**。
- 真实目录模式的价格是 GBP 历史最后观测单价（显示 £），不是当前售价；人民币预算会先澄清币种。
- 语音/视觉/手势的验收输入是「已转写 / 已识别 / 已确认」的结构化输入，真实麦克风、摄像头与
  真人手势需现场演示时另行确认；服务只绑回环地址、无账号认证，仅适合本机课程演示。
