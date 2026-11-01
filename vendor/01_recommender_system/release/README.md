# Oblivionis(dd2poolseed) 复现包

这是 Oblivionis 推荐系统的完整复现包。`Oblivionis` 是产品/模型名，`dd2poolseed` 是具体实现名；当前默认版本是线上综合分 **0.8060** 的 v2 修复版。

本包包含**两个版本**：

| 版本 | 线上综合分 | NDCG@10 | Recall@10 | GAUC | SHA256 |
|---|---:|---:|---:|---:|---|
| **v2（修复版，当前最好）** | **0.8060** | **0.8992** | 0.4728 | **0.8793** | `d924f90e7b3c661614738ecc6353882f6497508338fc85153b2dd555a4cd66a0` |
| v1（修复前存档） | 0.8049 | 0.8962 | 0.4736 | 0.8791 | `123816780ae366ee5e9bea43cd69d4188bc55fe8c12cd67c126b9923a25518e3` |

**v2 是审计后的重建版本**：修复了「引擎训练期穿越」与「特征—标签行错位」两处缺陷，
重训 H2 家族并在干净底座上重构融合权重，线上分数由 0.8049 提升到 **0.8060**。
完整过程见 `evidence/自审与修复报告.md`。v1 保留为修复前的存档，其复现路径不变。

## 最快的精确复现

```powershell
python -m pip install -r requirements-reproduce.txt
python reproduce_v2.py          # 重建 v2（0.8060）
python reproduce_oblivionis.py  # 重建 v1（0.8049）
python verify_oblivionis_package.py
```

建议使用 64 位 Python 3.13；依赖文件已经锁定本包实际验证通过的版本。两个脚本都只依赖 NumPy、pandas，都逐字节校验并与期望产物比对。

## 完整网页系统

复现包同时包含可独立启动的 Flask 前后端系统，默认加载修复版 v2（0.8060）：

```powershell
python -m pip install -r requirements-web.txt
python run_web.py
```

浏览器打开 `http://127.0.0.1:5000/`。三个演示账号会直接显示在登录页，密码均为
`demo123`。系统包含个性化“猜你喜欢”、分页、商品详情、相似商品、点击后实时插入、
喜欢/不喜欢、推荐会话与行为日志落库。运行前可执行：

```powershell
python verify_web_system.py
```

网页运行时不加载带原始交易记录的 93MB `engine.pkl`，而使用包内脱敏后的商品目录、
冻结 v2 排序和轻量 ItemCF 邻居索引，因此无需携带 `train.csv` 或原始交易数据库。
详细说明见 `web_system/README.md`。

## 算法配方

九个来源先分别做用户内百分位秩归一化。然后按用户分两段排序：

```text
Top-10 = .15·ours + .42·binary + .215·H94 + .215·H95
11名起 = .15·ours + .10·binary + .28·H94 + .10·H95
       + .07·H2L15 + .07·H2L7 + .07·H2L07s07
       + .07·H2L11s24 + .07·H2L15s24
```

先锁定 Top-10，再对剩余候选使用尾部配方，最终输出每个用户的名次分数 `0,-1,-2,...`。尾部原始权重和是 0.98；这是锁定脚本中的精确配方，整体缩放不会改变排序。

## 从训练数据完整重跑

`source/` 保存了完整版本化训练链的源码快照。大型训练数据、引擎模型和 v65 多窗口缓存没有在本快速包内重复复制，它们的逐文件大小与 SHA256 位于 `dependency_inventory.json`。

### 依赖放哪里

第二层复现需要一个"项目根"。**该目录的相对结构必须与下表一致** —— 所有路径都相对于 `--project-root` 指向的目录：

```text
<项目根>/
├── data/
│   ├── train.csv                              29,483,459 B   ← 必需（训练数据）
│   └── sample_candidates.csv                     704,940 B   ← 必需（本包 inputs/ 内有同名文件）
├── models/
│   └── competition_v11_season/              230.4 MB / 8 个    ← 必需（基线引擎）
│       ├── engine.pkl                         42,598,543 B
│       ├── rankers.pkl                       181,631,387 B
│       ├── item2vec.model                      2,035,362 B
│       ├── seasonal_state.pkl                  1,244,423 B
│       ├── user_vectors.npz                    1,095,666 B
│       ├── vector_index.npz                      907,040 B
│       ├── itemcf.npz                            866,912 B
│       └── model_meta.json                        11,281 B
├── outputs/
│   ├── competition_v65_capacity/
│   │   └── cache/                           248.4 MB / 3 个    ← 可选，缺失则自动重建
│   │       ├── train_win5.pkl                138,115,291 B
│   │       ├── train_win3.pkl                 82,895,689 B
│   │       └── full.pkl                       27,427,854 B
│   └── competition_v38_categorical/
│       └── production_v15_features.pkl        35,082,307 B    ← 必需（本包 inputs/layer2/ 已提供）
├── baseline/                            ← 必需（本包 source/code/baseline/ 已提供完整 18 个模块）
├── 0.7912的模型/
│   └── recommendation/                       17.1 MB / 52 个   ← 必需（外部特征定义 + EASE）
└── scripts/                                                   ← 训练脚本（本包 source/code/scripts/ 已提供快照）
```

> **两处易踩的坑（协作方核查后已修复）**
>
> 1. **`baseline/` 必须是完整模块集。** 包内快照早先只有 5 个文件，缺 `config.py` / `bpr.py` / `data.py`。
>    其中 `config.py` 是 `engine.py:38` 的**模块级导入**（`from .config import ROUTES`），
>    缺了 `import baseline` 直接失败；`bpr.py` / `data.py` 则被链上脚本显式 import。
>    现已补齐为 **18 个 .py**，并实测 `import baseline.{config,bpr,data,engine,competition,seasonal_features,features_v15}` 全部通过。
> 2. **`products.sqlite3` 不需要。** `engine.py:591` 的 sqlite 分支条件是与 `MODELS` 目录比较，
>    而 `MODELS = models/baseline`、链上用的是 `models/competition_v11_season` —— **分支不走**，实测确认。

**必需部分合计 ≈ 277 MB**（`data/` + `models/` + `0.7912的模型/`）；再加上可选的 v65 缓存共 **525.4 MB**。逐项核查结论见 `evidence/第二层依赖说明.md`；**从零搭建项目根跑通七步链的验证记录见 `evidence/第二层端到端验证记录.md`**（产出与锁定值逐字节相同，SHA `123816780ae366ee…`）。

### 校验依赖是否就位

完整清单（68 项的路径、字节数、SHA256）在 `dependency_inventory.json`。逐项比对：

```powershell
python check_dependencies.py --project-root D:\作业\行业工程实践\code
```

输出会列出每一类的通过数，以及任何缺失或哈希不符的文件。

### 运行

```powershell
cd D:\作业\行业工程实践\code
python D:/作业/行业工程实践/复现包_Oblivionis_dd2poolseed_20260911/retrain_oblivionis.py --project-root D:/作业/行业工程实践/code
```

脚本按顺序执行，任一步失败即中止：

| 步 | 脚本 | 耗时 | 产出 |
|---|---|---:|---|
| 0 | `experiment_v65_capacity.py` | 有缓存则跳过 | 5 窗口特征缓存 |
| 1 | `experiment_v67_gauc_objective.py` | 有产物则跳过 | 二分类目标来源 |
| 2 | `experiment_v95_O2_H2_fixed.py` | 538 s | H94 / H95 |
| 3 | `experiment_v102_H2_variants.py` | 85 s | H2L15/L31/L127/T1000/BIN/S2024 |
| 4 | `experiment_v104_smallcap.py` | 50 s | H2L7/L5/L3 + CA/CB |
| 5 | `experiment_v105_smallpool.py` | 30 s | L9/L11 + CC0~CC4 |
| 6 | `experiment_v106_headfocus.py` | 85 s | **DD2（最终产物）** |

**全程约 13 分钟**（复用 v65 缓存）。加 `--force-prerequisites` 会强制重跑第 0、1 步。跑完自动校验最终 SHA256，不符则报错退出。`v94` 作为特征列定义与训练模块被后续脚本复用。

## 目录

- `reproduce_oblivionis.py`：单文件精确装配与校验。
- `run_web.py`：启动完整网页系统（默认 `http://127.0.0.1:5000/`）。
- `verify_web_system.py`：验证登录、推荐、点击实时插入、喜欢反馈与 SQLite 落库。
- `web_system/`：完整前端、Flask 后端、轻量相似商品索引和运行时数据库定义。
- `check_dependencies.py`：校验第二层所需的 525 MB 外部依赖是否就位、哈希是否吻合。
- `retrain_oblivionis.py`：从训练数据完整重跑七个步骤的编排脚本。
- `inputs/`：候选集与九个冻结来源。
- `expected/`：锁定的最终提交。
- `source/code/`：训练、特征、信号和外部特征代码快照。
- `evidence/`：最终提交记录、完整推荐系统技术设计，以及**模型构建全流程**
  （`模型构建全流程.md` 讲这套模型怎么造出来；`模型构建全流程_单文件版.md` 为图表内嵌版；`figures/fig_arch_dd2.png` 为架构图）。
- `manifest.json`：包内不可变文件清单。
- `evidence/复现验证记录.md`：本包实际执行过的两层复现测试记录（含全部哈希与耗时）。
- `evidence/自审与修复报告.md`：**本次自审发现的两处训练链缺陷、修复方式与线上结果（0.8049 → 0.8060）**；`evidence/模型构建全流程.md` 顶部有对应的修正声明。
- `dependency_inventory.json`：未重复复制的大型依赖清单。

复现脚本生成的 `output/` 不在不可变清单内，可反复覆盖。
