# 部署、降级、风险与演示

> 阶段6 提交物（部署与风险文档），同时作为现场演示与答辩提纲。
> 数字口径：一切以 `outputs/` 下的实测文件为准，本文不写死通过率。

## 1. 依赖与启动

| 组件 | 依赖 | 说明 |
| --- | --- | --- |
| 模块三离线核心 | Python 3.10+ 标准库 | 契约、意图、会话、召回、解释、HTTP、验收脚本都不用第三方包 |
| 模块三 HTTP 服务 | 同上 | `python -B app.py`，只绑定 `127.0.0.1` |
| 模块一代码 | 仓库内 `vendor/01_recommender_system/`（或同级原项目） | 冻结 Flask 服务 + 冻结排序结果 + 商品目录；代码副本已在仓库里，clone 就能读 |
| 模块一接入（auto 模式） | 当前解释器装 Flask | 用于启动上面这份 `release/` 的冻结服务（无需 173 MB 模型 pickle） |
| 模块二代码 | 仓库内 `vendor/02_multimodal_ai_assistant/`（或同级原项目） | 只有 `src/` + `config/` 源码，供阅读 |
| 模块二接入（语音/视觉/手势） | 模块二自己的 `.venv` | 模型与重型依赖在模块二环境里；没有 venv 时媒体接口 503、保留手动输入 |

兄弟项目根目录规则：**同级优先，否则 vendor**（见 `config/settings.py::_module_root`）；
`IRA_MODULE_ONE_ROOT` / `IRA_MODULE_TWO_ROOT` 可显式指定。仓库没有带的东西：
模块一 173 MB 模型 pickle、61 MB 特征、72 MB 原始交易数据；模块二 `.venv` 与模型权重
（详见 [vendor/README.md](../vendor/README.md)）。

```powershell
# 默认 auto：自动接入同级模块一（需要 Flask）
python -m pip install -r ..\01_recommender_system\release\requirements-web.txt
python -B app.py

# 纯离线演示：不连模块一、不加载模型，只用 18 件合成商品
$env:IRA_PROFILE = 'demo'; python -B app.py

# 六阶段证据 + 回归测试
python -B scripts/run_all.py
python -B -m unittest discover -s tests -v
```

打开 <http://127.0.0.1:8080>。端口被占用时 `$env:IRA_PORT='8081'`。
关键路径全部来自 `config/settings.py`，环境变量：`IRA_PROFILE`、`IRA_CATALOG`、`IRA_OUTPUT_DIR`、
`IRA_PORT`、`IRA_RECOMMENDER_URL`、`IRA_RECOMMENDER_USERNAME`、`IRA_RECOMMENDER_PASSWORD`、`IRA_TIMEOUT`、
`IRA_MODULE_ONE_ROOT`、`IRA_MODULE_ONE_PORT`、`IRA_MODULE_TWO_ROOT`、`IRA_MODULE_TWO_PYTHON`、
`IRA_MODULE_TWO_TEXT_PORT`、`IRA_MODULE_TWO_VISION_PORT`、`IRA_MEDIA_TIMEOUT`。

## 2. 自动接入失败与回退

只有「找得到模块一发布包（同级 `01_recommender_system/release` 或仓库内 `vendor/01_recommender_system/release`）、
且未显式设置 `IRA_PROFILE=demo`/`IRA_CATALOG`/`IRA_RECOMMENDER_URL`」时，`app.py` 才会启动模块一：
- 有 `development/data/train.csv`（原项目）→ 每次启动由两份 CSV 重新生成 `data/module_one_catalog.json`；
- 没有原始交易数据（仓库 vendor 副本）→ 直接用仓库里已生成的 `data/module_one_catalog.json`；
然后启动冻结 Flask 子进程（5001）。

- **有模块一但启动失败**（缺 Flask、5001 被占用、启动超时）：`prepare()` 捕获异常、关闭子进程，
  回退到 `offline_demo_settings()`（固定 `data/catalog.json`、清空远端凭据），stderr 打印
  「警告：自动接入模块一失败，已回退离线演示（<异常类型>: <异常信息>）。」，主服务继续可用。
- **找不到模块一（同级和 vendor 都没有），或显式 `IRA_PROFILE=demo`**：正常离线启动，不打印回退警告。
- **显式 `IRA_RECOMMENDER_URL` / `IRA_CATALOG`**：按显式配置运行，不自动启动模块一。
- **本项目自身端口被占用**：直接抛错退出（不属于模块一接入失败），用 `IRA_PORT` 换端口。

模块二不可用（`.venv` 缺失、模型不可用、超时）不会让主服务退出：媒体接口返回明确错误码
（`UPSTREAM_UNAVAILABLE`/`UPSTREAM_TIMEOUT`/`INVALID_PERCEPTION`），网页保留手动转写、标签与手势入口。

## 3. 降级矩阵（故障注入与观察点）

| 场景 | 行为 | 观察点 / 证据 |
| --- | --- | --- |
| 找不到模块一（同级和 vendor 都没有）或 `IRA_PROFILE=demo` | 用 `data/catalog.json` 合成目录正常启动，不打印回退警告 | `GET /api/health` 的 `mode=offline_demo` |
| 有模块一但启动失败 | stderr 打印回退警告，改用合成目录继续服务 | 启动 stderr/stdout |
| 推荐上游超时/不可用 | 按 popularity 离线召回，`degraded=true`，`warnings[0].code=UPSTREAM_TIMEOUT` | `outputs/stage03_demo.json`、`stage06_acceptance.json` |
| 商品ID与目录无交集 | `INVALID_UPSTREAM` 告警并降级，不编造价格或属性 | `warnings` |
| 模块二不可用 | 语音/视觉/手势接口 503 + 错误码，页面保留手动输入 | `GET /api/health` 的 `module_two.configured=false` |
| 视觉置信度 < 0.7 | 忽略标签并记录 `ignored_labels`，不进入约束 | `outputs/stage02_context_cases.jsonl` |
| 硬约束无候选 | `status=empty`，不放宽预算/品类/颜色，提示调整或撤销 | `outputs/stage03_demo.json` |
| 解释证据缺失/属性矛盾 | 停止解释、清空引用，返回 `INSUFFICIENT_EVIDENCE`/`EVIDENCE_MISMATCH` | `outputs/stage04_abnormal_explanations.md` |
| 本项目端口被占用 | 报错退出（不属于模块一接入失败，不回退），换 `IRA_PORT` | 启动 traceback |

## 4. 风险登记表

| 风险 | 影响 | 当前措施 | 后续验证 |
| --- | --- | --- | --- |
| 前两个模块环境不同 | 模型服务不可用 | 模块二独立 venv 子进程；模块一独立 Flask 服务；都保留离线降级 | 物理设备与并发现场验证 |
| 上游缺价格/颜色 | 预算过滤无依据 | 只对同 ID 对齐；缺价排除预算筛选、缺色不猜测 | 增补有依据的现价与属性 |
| 自然语言表达复杂 | 规则漏识别 | 未解析预算追问；支持显式 `slots` | 扩充口语、否定、组合表达题集 |
| 视觉识别错误 | 错误筛选 | 白名单标签 + 阈值 0.7 + 冲突拒绝更新；自报评分明确标注 | 在真实识别集标定置信度 |
| 会话串扰 | 偏好误用 | `(user_id, session_id)` 隔离 + TTL + 结束清理 | 多用户认证与并发压力验证 |
| 模型幻觉 | 错误解释 | 解释只读属性与证据包，引用必须属于本商品 | 更大目录下验证引用支持关系 |
| 日志泄露隐私 | 用户输入暴露 | 不记录请求体；原图/音频不落盘；样例只用合成数据 | 对导出 JSON 人工审查 |
| 服务阻塞 | 本地交互等待 | 上游超时、请求大小上限、模型串行 + 超时回收 | 模型队列与取消机制 |
| 合成样例过拟合 | 高通过率被误读为真实效果 | 报告明确样例范围与延迟口径 | 真实用户查询测试 |

六维度（隐私、安全、算法偏差、知识产权、法律文化、环境）：
隐私与数据安全 —— 会话只存内存、媒体不落盘、导出 JSON 由导出者审查；
密钥与服务安全 —— 只绑回环、无公网、演示账号是模块一发布页公开凭据、密码不写进文档；
算法偏差 —— 多路召回保留来源与分数明细、价格/颜色缺失明确标注、通过率只代表固定样例；
知识产权 —— 只通过公开接口复用相邻模块，数据与模型来源在 `data/module_one_catalog.json`
的 `dataset_reference/sources`（含 SHA256）标注，提交包不含模型权重与大数据集；
法律文化 —— 界面标注“£历史单价，不是当前售价”，人民币预算先澄清币种、不自动换算；
环境 —— 模型按需加载、串行推理、超时回收进程树，离线核心只用标准库。

## 5. 十分钟现场演示

1. **0—1 分钟**：说明边界——auto 模式接入 2,208 件真实商品与 v2 冻结排序（GBP 历史单价），
   `IRA_PROFILE=demo` 时为 18 件合成商品；展示目录结构（`assistant/` 十来个文件、`scripts/01~06`）。
2. **1—2 分钟**：`python -B scripts/run_all.py`，打开 `outputs/acceptance_summary.json`
   说明各阶段用例数与通过情况（数字以 JSON 为准，不口头编）。
3. **2—3 分钟（阶段2）**：`python -B scripts/02_context_test.py`，打开
   `outputs/stage02_context_cases.jsonl` 与 `stage02_error_cases.md`，演示三个行为：
   预算“三百元”未解析 → 追问；同轮颜色冲突 → 拒绝更新；撤销 → 恢复上一步。
4. **3—4 分钟（阶段1/3）**：打开 `stage01_contract_report.json` 说明统一事件、trace_id 与错误码；
   `python -B scripts/03_orchestrator_demo.py`，展示正常/空集/上游超时三类 trace 与确定性复现。
5. **4—6 分钟**：运行 `app.py`，点“为我推荐商品”，切换模块一用户观察排序变化；
   输入“推荐红色杯子，预算10英镑以内”，展示预算、颜色、证据、trace_id。
6. **6—7 分钟**：切“语音”录入“红色杯子，预算10英镑以内”；新会话切“视觉”上传图片并补预算；
   没有设备时用手动转写/标签，说明原因。
7. **7—8 分钟**：点赞商品展示排名提升；切“手势”开启摄像头做点赞/倒拇指（无设备用手动）；
   新会话展示反馈隔离；打开 `stage05_performance_summary.md` 说明前后排名/分数变化与无变化案例。
8. **8—9 分钟**：勾选两件商品比较，展开证据与分数明细，导出当前 trace；
   打开 `stage04_abnormal_explanations.md` 解释“无证据不编造/字段缺失/价格不一致”。
9. **9—10 分钟**：新会话设预算 0.01 英镑展示空结果不放宽硬约束；输入人民币预算展示币种澄清；
   最后指出每位组员负责的模块（见 `docs/五人分工.md`）。

## 6. 五分钟答辩问答准备

- **为什么用适配器？** 三个项目只交换稳定契约，不读对方 GUI 或内部变量，任何一方可独立替换/降级。
- **四类契约是什么？** `InteractionEvent`（统一输入）、`RecommendationRequest`（推荐请求）、
  `RecommendationItem`（结果）、`EvidencePack`（可解释证据）；`trace_id` 贯穿全链路，失败带错误码。
- **如何防止高分商品突破预算？** 先硬过滤再打分，空集不放宽约束（`outputs/stage03_demo.json`）。
- **如何证明解释可信？** 解释只读 `EvidencePack`，引用必须属于本商品且属性一致，缺证据就降级。
- **为什么点赞有效？** 当前会话反馈权重 ±2.5，重复反馈覆盖不累积，分数明细可见。
- **模型不可用怎么办？** 明确错误码 + 离线目录召回，硬约束不变，trace 完整保留。
- **通过率代表真实效果吗？** 不代表；只是固定样例上的契约/流程验收，没有线上点击率或模型准确率结论。
- **五人怎么分工？** 见 `docs/五人分工.md`，每人一个阶段、一组文件、一组证据。

## 7. 演示材料与提交

- 当前**没有演示视频**：本文 + 运行截图（`outputs/browser_preview.png` 等）作为现场演示材料。
- 提交包要求见 [提交清单](submission_checklist.md)：`小组号-项目名` 目录、源代码、`requirements.txt`、
  `README.md`、每人一份 PDF 报告；不提交 `runtime/`、缓存、`.env`、模型权重、个人音视频。
  打包前可检查是否混入音视频与缓存：
  `Get-ChildItem -Recurse -File | Where-Object { $_.Extension -in '.wav','.mp4','.pyc' }`。
