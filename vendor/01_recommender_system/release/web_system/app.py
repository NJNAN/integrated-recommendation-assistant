"""模块一（推荐系统）的 Flask 服务：登录、推荐分页、商品详情、相似商品、反馈。

本仓库这份是**只读副本，为了讲解补充了中文注释**（逻辑与上游原项目一致，AST 已对比过）。
模块三（阶段3）只调用其中三个接口：
    POST /api/auth/login        登录，换 cookie session
    GET  /api/recommendations   按用户拿推荐（分页）
    GET  /api/health            判断服务是否就绪（model_ready）

其余接口（商品详情、相似商品、点击/反馈）模块三没有调用，知道它们存在即可。
阅读约定：每个函数都写了「参数 / 返回 / 怎么做」。
"""
from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import uuid

from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash

from web_system.service import DEMO_ACCOUNTS, RecommendationService


ROOT = Path(__file__).resolve().parent        # .../release/web_system
PACKAGE_ROOT = ROOT.parent                    # .../release（冻结排序结果在它的 expected/ 下）


def create_app(test_config: dict | None = None, service_instance=None) -> Flask:
    """创建 Flask 应用：配置四个数据路径、注册路由、懒加载推荐服务。

    参数：
        test_config (dict | None): 测试用，覆盖下面 app.config 里的任意键；正常传 None。
        service_instance (RecommendationService | None): 直接注入一个已建好的服务实例；
            正常传 None，等第一次真正用到时再懒加载（避免启动就加载 736 个用户的排序）。
            模块三的 integrations/module_one_server.py 就是 `create_app()` 之后直接 run。

    返回：
        Flask: 应用对象（调用方负责 app.run）。

    怎么做的：
        1. 配置 SECRET_KEY（cookie 签名）、cookie 安全选项；
        2. 配置四个数据路径（见下表），全部可被 test_config 覆盖；
        3. 定义一个内部函数 `service()`：第一次调用时才 new RecommendationService；
        4. 注册路由（见文件后半部分）；
        5. 返回 app。

    app.config 的四个关键路径：
        SCORE_PATH      release/expected/submission_v2_R1_head48.csv   冻结排序结果
        CATALOG_PATH    web_system/data/catalog.csv                    2208 件商品
        NEIGHBORS_PATH  web_system/data/item_neighbors.json            相似商品索引
        DATABASE_PATH   web_system/runtime/oblivionis_web.sqlite3      会话/反馈数据库
                        （模块三用环境变量 OBLIVIONIS_DATABASE_PATH 把它改到自己的 runtime/ 下）
    """
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config.update(
        SECRET_KEY=os.getenv("OBLIVIONIS_SECRET_KEY", "oblivionis-local-demo-change-me"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("OBLIVIONIS_COOKIE_SECURE", "0") == "1",
        SCORE_PATH=PACKAGE_ROOT / "expected" / "submission_v2_R1_head48.csv",
        CATALOG_PATH=ROOT / "data" / "catalog.csv",
        NEIGHBORS_PATH=ROOT / "data" / "item_neighbors.json",
        DATABASE_PATH=Path(os.getenv("OBLIVIONIS_DATABASE_PATH", str(ROOT / "runtime" / "oblivionis_web.sqlite3"))),
    )
    if test_config:
        app.config.update(test_config)
    state = {"service": service_instance}

    def service() -> RecommendationService:
        """懒加载推荐服务（只在第一次请求时创建）。

        返回：
            RecommendationService: 全局唯一的服务实例（存在闭包 state 里）。

        为什么懒加载：创建服务要读 1.3 MB 排序 CSV + 0.15 MB 商品表 + 建 sqlite 表，
        启动就做会让「服务是否就绪」的健康检查变慢；懒加载后第一次请求承担这个成本。
        """
        if state["service"] is None:
            state["service"] = RecommendationService(
                Path(app.config["SCORE_PATH"]),
                Path(app.config["CATALOG_PATH"]),
                Path(app.config["NEIGHBORS_PATH"]),
                Path(app.config["DATABASE_PATH"]),
            )
        return state["service"]

    def ok(data=None, status=200):
        """统一成功响应：`{code:0, message:'success', data:...}`。

        参数：
            data: 要返回的数据（任意可 JSON 序列化的对象）。
            status (int): HTTP 状态码，默认 200。
        返回：
            (Response, int): Flask 的 jsonify 结果与状态码。
        """
        return jsonify({"code": 0, "message": "success", "data": data}), status

    def fail(message, status=400):
        """统一失败响应：`{code:status, message:..., data:None}`。

        参数：
            message (str): 中文错误说明。
            status (int): HTTP 状态码，默认 400。
        返回：
            (Response, int)。
        """
        return jsonify({"code": status, "message": message, "data": None}), status

    def require_auth(view):
        """装饰器：要求已登录，否则返回 401「请先登录」。

        参数：
            view: 被装饰的视图函数。
        返回：
            包装后的视图函数（先检查 session 里有没有 user）。
        """
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user"):
                return fail("请先登录", 401)
            return view(*args, **kwargs)
        return wrapped

    def int_arg(name: str, default: int, minimum: int, maximum: int) -> int:
        """从 query string 取一个整数参数并做范围校验。

        参数：
            name (str): 查询参数名，例如 'cursor'、'limit'。
            default (int): 没传时的默认值。
            minimum (int): 允许的最小值（含）。
            maximum (int): 允许的最大值（含）。
        返回：
            int: 校验通过的整数。
        异常：
            ValueError: 不是整数、或超出范围（调用方会转成 400）。
        """
        try:
            value = int(request.args.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
        return value

    @app.get("/")
    def index():
        """模块一自己的前端首页（模块三不使用）。"""
        return render_template("index.html")

    @app.get("/health")
    @app.get("/api/health")
    def health():
        """健康检查：三个数据文件都存在就算就绪。

        参数：无。
        返回：
            {code:0, data:{status:'ok', model_ready: bool}}。
        模块三怎么用：`module_one.py::ModuleOneService.start()` 轮询它，
        直到 `data.model_ready` 为 True 才认为服务可用（最多等 20 秒）。
        """
        ready = all(Path(app.config[key]).exists() for key in ("SCORE_PATH", "CATALOG_PATH", "NEIGHBORS_PATH"))
        return ok({"status": "ok", "model_ready": ready})

    @app.get("/api/system")
    def system_info():
        """模型/数据版本信息（模块三不使用）：模型名、指标、用户数、候选数等。"""
        return ok(service().system_info())

    @app.get("/api/auth/demo-accounts")
    def demo_accounts():
        """把三个演示账号直接抛出来（模块三不使用；账号本来就写在 service.py 里）。"""
        return ok([
            {"username": username, "display_name": display_name, "password": password}
            for username, _, display_name, password in DEMO_ACCOUNTS
        ])

    @app.post("/api/auth/login")
    def login():
        """登录：校验账号密码，成功后把用户写进 cookie session。

        参数（JSON body）：
            username (str): 演示账号（northstar / paperplane / greengate），大小写不敏感；
            password (str): 演示密码（发布页公开的 demo123）。
        返回：
            成功：{code:0, data:{username, user_id, display_name}}；
            失败：401 {code:401, message:'账号或密码不正确'}。
        怎么做的：
            1. 用小写 username 去 `service().account()` 查账号行（含密码 hash）；
            2. `check_password_hash` 比对；
            3. `session.clear()` 后写入 user 和空的 hidden_items（点过“不喜欢”的商品列表）。
        模块三怎么用：`HTTPRecommender.recall` 先 login，再核对返回的 user_id 与请求一致。
        """
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip().lower()
        row = service().account(username)
        if not row or not check_password_hash(row[3], str(payload.get("password", ""))):
            return fail("账号或密码不正确", 401)
        session.clear()
        session["user"] = {"username": row[0], "user_id": row[1], "display_name": row[2]}
        session["hidden_items"] = []
        return ok(session["user"])

    @app.post("/api/auth/logout")
    def logout():
        """登出：清空 cookie session（模块三不使用）。"""
        session.clear()
        return ok({"logged_out": True})

    @app.get("/api/auth/status")
    def auth_status():
        """查询当前是否登录（模块三不使用）。"""
        return ok({"authenticated": bool(session.get("user")), "user": session.get("user")})

    @app.get("/api/auth/me")
    @require_auth
    def me():
        """返回当前登录用户（模块三不使用）。"""
        return ok(session["user"])

    @app.get("/api/recommendations")
    @require_auth
    def recommendations():
        """按当前登录用户返回推荐，分页。

        参数（query string）：
            cursor (int): 从第几条开始，0~100000，默认 0 —— 模块三的翻页游标；
            limit (int): 一次返回几条，1~24，默认 12 —— 模块三固定传自己的 max_top_k=24。
        返回：
            {code:0, data:{items:[{item_id, name, description, category,
                                  score(0~1), source_type, reason_code, reason, session_id}],
                           session_id, next_cursor, has_more, model_version}}。
        怎么做的：
            1. 校验 cursor/limit；
            2. 从 cookie session 取 user_id 和 hidden_items（避免返回用户点过“不喜欢”的商品）；
            3. 调 `service().recommendations(user_id, cursor, limit, exclude)`；
            4. 包成统一格式返回。
        模块三怎么用：`HTTPRecommender.recall` 循环调它，用 has_more/next_cursor 翻页，
        只取 item_id 和 score。
        """
        try:
            cursor = int_arg("cursor", 0, 0, 100000)
            limit = int_arg("limit", 12, 1, 24)
        except ValueError as exc:
            return fail(str(exc))
        return ok(service().recommendations(
            session["user"]["user_id"], cursor, limit, set(session.get("hidden_items", []))
        ))

    @app.get("/api/items/<item_id>")
    @require_auth
    def item_detail(item_id):
        """商品详情 + 相似商品前 6 个（模块三不使用；我们用自己目录里的属性）。"""
        return ok({"item": service().product(item_id), "similar": service().similar(item_id, 6)})

    @app.get("/api/items/<item_id>/similar")
    @require_auth
    def similar(item_id):
        """相似商品列表（模块三不使用）。

        参数：
            item_id (str): 路径参数，商品 ID；
            limit (int): query，1~12，默认 6；
            exclude (str): query，逗号分隔的要排除的商品 ID。
        返回：
            {code:0, data:{item_id, items:[...]}}。
        """
        try:
            limit = int_arg("limit", 6, 1, 12)
        except ValueError as exc:
            return fail(str(exc))
        exclude = set(filter(None, request.args.get("exclude", "").split(",")))
        return ok({"item_id": item_id, "items": service().similar(item_id, limit, exclude)})

    @app.post("/api/events/click")
    @require_auth
    def click_event():
        """记录一次点击，并顺手返回 4 个相似商品（模块三不使用）。

        参数（JSON body）：
            item_id (str): 被点击的商品，必填；
            idempotency_key (str): 幂等键，不传就生成 UUID；
            visible_item_ids (list): 当前页可见商品，返回相似商品时会排除。
        返回：
            {code:0, data:{event_recorded, insert_strategy, anchor_item_id, items}}。
        """
        payload = request.get_json(silent=True) or {}
        item_id = str(payload.get("item_id", "")).strip()
        if not item_id:
            return fail("item_id 不能为空")
        payload.update({
            "item_id": item_id,
            "feedback_type": "click",
            "idempotency_key": payload.get("idempotency_key", str(uuid.uuid4())),
        })
        try:
            service().feedback(session["user"]["user_id"], payload)
            exclude = set(map(str, payload.get("visible_item_ids", [])))
            items = service().similar(item_id, 4, exclude)
        except Exception as exc:
            return fail(f"点击行为处理失败：{exc}")
        return ok({"event_recorded": True, "insert_strategy": "frontend_inline", "anchor_item_id": item_id, "items": items})

    @app.post("/api/feedback")
    @require_auth
    def feedback():
        """记录喜欢/不喜欢；不喜欢会加入 hidden_items，后续推荐里排除（模块三不使用）。

        参数（JSON body）：
            item_id (str): 商品 ID，必填；
            feedback_type (str): 'like' 或 'dislike'，必填；
            idempotency_key (str): 幂等键，不传就生成。
        返回：
            {code:0, data:{recorded:true, feedback_type}}。
        说明：模块三的点赞/点踩只写自己的会话偏好，**不调用这个接口**。
        """
        payload = request.get_json(silent=True) or {}
        item_id = str(payload.get("item_id", "")).strip()
        feedback_type = payload.get("feedback_type")
        if not item_id or feedback_type not in {"like", "dislike"}:
            return fail("item_id 与 feedback_type(like/dislike) 必填")
        payload.update({"item_id": item_id, "idempotency_key": payload.get("idempotency_key", str(uuid.uuid4()))})
        try:
            service().feedback(session["user"]["user_id"], payload)
        except Exception as exc:
            return fail(f"反馈记录失败：{exc}")
        if feedback_type == "dislike":
            session["hidden_items"] = list(dict.fromkeys([*session.get("hidden_items", []), item_id]))[-100:]
        return ok({"recorded": True, "feedback_type": feedback_type})

    @app.errorhandler(404)
    def not_found(_):
        """404 也返回统一 JSON 格式，而不是 Flask 默认的 HTML。"""
        return fail("接口不存在", 404)

    @app.errorhandler(405)
    def method_not_allowed(_):
        """405 同理，返回统一 JSON。"""
        return fail("请求方法不允许", 405)

    return app


app = create_app()
