# 课程提交清单（人工事项）

对应 `docs/材料提交规范及内容要求.txt` 与 `docs/《行业工程实践》普本学生实训报告模板.docx`。
代码、脚本、证据由仓库自动产出；本文件只列**必须由组员本人完成**的事项，不得代填、不得编造。

## 1. 打包步骤

```powershell
# 1) 停掉正在运行的服务，清理运行产物
Remove-Item -Recurse -Force runtime -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File -Include *.pyc | Remove-Item -Force

# 2) 重新生成六阶段证据（确保 outputs 是最新实测）
python -B scripts/run_all.py
python -B scripts/test_report.py

# 3) 复制成学校要求的目录名再压缩
Copy-Item -Recurse . "..\小组号-项目名"
Compress-Archive "..\小组号-项目名" "..\小组号-项目名.zip"
```

- [ ] 提交目录命名为 **`小组号-项目名`**（当前目录名是脚手架名 `03_integrated_recommendation_assistant`，需重命名或复制后改名）。
- [ ] 包含全部源代码：`app.py`、`assistant/`、`integrations/`、`config/`、`web/`、`tests/`、`scripts/`、
      `data/`、`outputs/` 证据、`docs/` 材料、`requirements.txt`、`README.md`。
- [ ] 不包含：`runtime/`、`__pycache__/`、`*.pyc`、`.env`、模型权重、个人音视频、`.work/`。
- [ ] 打包前检查是否混入音视频/缓存：
      `Get-ChildItem -Recurse -File | Where-Object { $_.Extension -in '.wav','.mp4','.pyc','.sqlite3' }`

## 2. 实训报告（每人一份 PDF）

- [ ] 每人独立填写 `docs/《行业工程实践》普本学生实训报告模板.docx`，提交前删除模板里的“（填写提示：…）”，
      在 Word 中更新目录；导出 **`小组号-成员名-实训报告.pdf`**，一人一份。
- [ ] 封面字段：学年学期、课程名称/代码、项目名称、专业、班级、组别、姓名、学号、指导教师、总分、学院、年月。
- [ ] 表1-1 需求与验收指标、表1-2 项目模块与成果、表1-3 成员分工；表2-1/2-2 硬件与版本；
      第3/4 章引用同级模块一/模块二报告（标注出处）；第5 章写本项目核心（架构、契约、召回、解释、反馈、验收）；
      第6 章六维度风险（可直接引用 `docs/deployment.md` 第4节）；第7 章测试与验收；
      第8 章 ≥800 字真实经历；第9 章参考资料（数据集与模型来源）；附录表A-1 指标追溯索引。
- [ ] 第5 章数字以 `outputs/` 为准：57 条固定样例（阶段02 要求 ≥20 条、通过率 ≥80%）、
      回归测试数量见 `outputs/test_results.json:tests_run`；失败案例按阶段引用
      `docs/stage_summaries.md` 的证据路径。
- [ ] 表7-2 的“Git 提交”行：当前目录**没有 Git 仓库**，改用实验日志/结构化输出/截图说明；
      如学校要求 Git，先 `git init` 并保留提交历史。
- [ ] 成绩评定表由指导教师填写评语、等级、签字与日期。

## 3. 成员信息与分工

见 [五人分工](五人分工.md)：每人一个阶段、一组文件、一组证据。要求：

- [ ] 与报告表1-3、表7-1/7-2/7-3 一致；
- [ ] 答辩时每人 1—2 分钟独立讲本人模块与跨模块协议，不能照读或由他人代替；
- [ ] 不把清单或 AI 生成记录直接归属为某位成员的亲历工作。

## 4. 相邻小组复现记录（每阶段一次）

六份任务书“步骤4”都要求“相邻小组按 README 重跑一个用例并记录问题”。请邀请相邻小组/助教逐阶段记录：

| 项 | 内容 |
| --- | --- |
| 阶段 | 阶段01/02/03/04/05/06（每次一份） |
| 复现人/小组 | （其他小组真实信息，不得代签） |
| 日期与机器 | （实际日期；注明是 auto 模式还是 `IRA_PROFILE=demo`） |
| 执行命令 | 例如 `python -B scripts/02_context_test.py`，或 `python -B app.py` + 一个用例 |
| 预期/实际结果 | （退出码、输出文件、界面现象） |
| 遇到的问题 | （无则写“无”，有则记录错误信息与定位） |
| 修复与复测 | （说明是否修复、复测命令与结果；只改文字不算修复） |

## 5. 物理设备与演示材料

- [ ] 物理设备实测（有设备的成员）：麦克风（16kHz 单声道 16 位 WAV）、摄像头、真人手势，
      记录日期/设备/现象/结果。`outputs/stage06_acceptance.json` 的输入是“已转写/已识别/已确认”
      的结构化数据，**不代表物理设备已现场测试**（`scope.real_models_verified=false`）。
- [ ] 10 分钟演示 + 5 分钟答辩，按 `docs/deployment.md` 第5、6 节彩排。
- [ ] 演示视频（如学校要求）：按同一提纲录制；当前没有视频文件，用文档 + 运行截图作为演示材料。
- [ ] 演示环境：auto 模式需当前解释器装 Flask 并能发现同级模块一；否则用 `IRA_PROFILE=demo`；
      语音/视觉/手势需模块二 `.venv` 与物理设备，无设备时用手动输入并明确说明。

## 6. 隐私、密钥与数据

- [ ] 提交包与报告不出现密码/密钥/`.env`/真实用户音视频/个人敏感信息；模块一演示账号是发布包公开凭据，
      报告中也不记录密码。
- [ ] 网页“导出当前 trace”的 JSON 可能含主动输入与识别文本，提交前逐份审查，或只提交 `outputs/` 中的验收证据。
- [ ] 服务只绑 `127.0.0.1`、无账号认证，仅用于本机课程演示，不得公网部署。

## 7. 最终核对

- [ ] 在干净目录里重新跑一次 `python -B scripts/run_all.py` 与
      `python -B -m unittest discover -s tests`，全部通过。
- [ ] `python -B app.py` 能打开页面、`GET /api/health` 正常、四个输入模式可用（无设备时手动输入）。
- [ ] 每人 PDF 就位且命名正确；成员信息、分工、签字页已填写。
- [ ] 相邻小组复现记录 6 份、物理设备记录、演示视频（如需）已归档。
