# Oblivionis 完整网页系统

这是复现包内可独立启动的 Flask 前后端演示，默认使用修复版 v2 排序（线上综合分 0.8060）。

## 启动

```powershell
python -m pip install -r requirements-web.txt
python run_web.py
```

浏览器打开 `http://127.0.0.1:5000/`。页面会列出三个演示账号，密码均为 `demo123`。

Linux 服务器建议使用生产 WSGI 服务：

```bash
python3 -m pip install -r requirements-deploy.txt
gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 --access-logfile - web_system.app:app
```

## 功能

- 三个用户账号登录，各自展示不同的 Oblivionis 推荐排序；
- “猜你喜欢”分页加载与换一批；
- 点击商品展示详情和相似商品；
- 点击后把相似商品实时插入推荐流；
- 喜欢/不喜欢反馈即时响应并写入 SQLite；
- 推荐会话、曝光和反馈均可追溯；
- `/api/health` 提供健康检查。

## 数据边界

网页运行时只包含冻结推荐分数、商品展示目录和脱敏后的 ItemCF 邻居索引，不包含
`train.csv`、原始交易明细或带训练数据的 pickle。首次启动会在 `web_system/runtime/`
自动创建 SQLite 行为库；该目录可删除后重新生成。
