# vendor/ —— 兄弟项目的只读副本

> 这个目录是为了**让仓库自己能读通**：模块三的代码会调用「模块一 推荐系统」和
> 「模块二 多模态助手」，但它们是同级目录里的独立项目。如果仓库里没有它们的代码，
> 别人 clone 下来就只看到一堆 `release/web_system`、`.venv` 之类的路径却不明白是什么。

## 1. 里面是什么

### `vendor/01_recommender_system/`（模块一，推荐系统）

| 路径 | 内容 | 为什么带 |
| --- | --- | --- |
| `release/web_system/` | 冻结的 Flask 服务源码：`app.py`（路由/登录）、`service.py`（排序与反馈）、`database.py`、`templates/`、`static/`、`tests/`、`README.md` | 模块三通过公开 HTTP 接口调它，是“真实推荐能力”的本体 |
| `release/web_system/data/catalog.csv` | 2,208 件商品表（155 KB） | 启动服务必需 |
| `release/web_system/data/item_neighbors.json` | 相似商品索引（997 KB） | 服务返回“相似商品”必需 |
| `release/expected/submission_v2_R1_head48.csv` | Oblivionis v2 冻结排序结果（1.33 MB） | 服务的推荐分数来源 |
| `release/source/` | 模型源码（baseline/召回/特征/优化脚本等 54 个文件） | 想了解“排序是怎么训出来的”可以读 |
| `release/*.json / *.md / requirements-*.txt / verify_*.py` | 版本、依赖、校验与复现脚本 | 追溯数据来源与版本 |

**没有带**（体积太大，见 `release/manifest.json` 与 `development/` 原项目）：
`release/第二层依赖包_20260911/`（约 500 MB，含 173 MB `rankers.pkl`）、`release/inputs/`（61 MB 特征）、
`release/evidence/`、`development/data/`（72 MB 原始交易，含 28 MB `train.csv`）、`.venv/`、模型权重。

### `vendor/02_multimodal_ai_assistant/`（模块二，多模态助手）

| 路径 | 内容 |
| --- | --- |
| `src/` | 40 个源文件：`asr.py`（SenseVoice 语音识别）、`vl_model.py`（视觉模型）、`gesture.py`、`human_detector.py`、`knowledge_base.py`、`rag.py`、`llama_backend.py`、GUI 与音频线程等 |
| `config/settings.py` | 模型路径与端口配置（`MODEL_CONFIG` / `LLAMA_CONFIG`） |
| `main.py`、`README.md`、`RESULTS.md`、`RESUME.md`、`NEGATIVE-RESULTS.md` | 入口、说明与实验结果 |
| `requirements*.txt`、`pyproject.toml` | 依赖清单 |
| `启动助手.bat/.ps1`、`start_assistant.bat/.ps1` | 原项目的启动脚本 |

**没有带**：`.venv/`（数 GB 的重型依赖）、`models/`（本地模型权重）、`data/`、`outputs/`、`assets/`、
40 多个 `.pytest_*` 测试残留目录、文档压缩包。模块二只有在原项目的 `.venv` 里才能真跑。

## 2. 模块三怎么找到它们

`config/settings.py::_module_root()` 的规则：**同级目录优先，没有再退回本目录**。

```text
开发机：../01_recommender_system 存在      → 用原项目（可以跑完整模型/数据）
别人 clone：没有同级目录                  → 用 vendor/01_recommender_system（代码可见、服务可起）
```

- 模块一：`assistant/module_one.py::integrated_settings()` 会检查
  `release/web_system/data/catalog.csv` 与 `release/expected/submission_v2_R1_head48.csv`。
  - 有 `development/data/train.csv`：现场重算 `data/module_one_catalog.json`；
  - 没有（vendor 情况）：直接使用仓库里已生成的 `data/module_one_catalog.json`（内容与原始数据一致，含 sha256 来源）。
- 模块二：`assistant/module_two.py` 需要 `module_two_root/.venv/Scripts/python.exe`。
  vendor 里没有 venv，所以**语音/视觉/手势会返回 503 并保留手动输入**——这是预期的降级。

## 3. 能跑什么、不能跑什么

| 模式 | 条件 | 结果 |
| --- | --- | --- |
| `IRA_PROFILE=demo` | 只要标准库 | 18 件合成商品，全部功能可演示（语音/视觉/手势走手动输入） |
| 默认 auto + 本机装 Flask | 用 vendor 模块一 | 2,208 件真实商品 + 冻结排序，`mode=hybrid`，服务由模块三启动在 5001 |
| auto + 模块二 .venv | 需要在原项目环境里 | 真实语音/视觉/手势识别 |
| auto + 没有 Flask | 无 | 自动回退离线演示，并在 stderr 打印回退警告 |

## 4. 使用约定

- **只读**：不要直接改 vendor 里的文件；上游项目改完后重新复制（保持两边一致）。
- 复制来源与时间：2026-11-01 从同级 `01_recommender_system`、`02_multimodal_ai_assistant` 复制；
  模块一 84 个文件 / 4.4 MB，模块二 55 个文件 / 0.36 MB。
- 数据来源：模块一使用 UCI Online Retail II 公开数据集（见
  `release/web_system/README.md` 与 `data/module_one_catalog.json` 的 `dataset_reference`）；
  模型为 Oblivionis v2（`release/v2_meta.json`）。
- 许可证与版权：仅用于课程作业演示；第三方数据集与模型遵循其原始许可。
