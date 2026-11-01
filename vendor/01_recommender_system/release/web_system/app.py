"""Flask application for the packaged Oblivionis recommendation system."""
from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import uuid

from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash

from web_system.service import DEMO_ACCOUNTS, RecommendationService


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent


def create_app(test_config: dict | None = None, service_instance=None) -> Flask:
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
        if state["service"] is None:
            state["service"] = RecommendationService(
                Path(app.config["SCORE_PATH"]),
                Path(app.config["CATALOG_PATH"]),
                Path(app.config["NEIGHBORS_PATH"]),
                Path(app.config["DATABASE_PATH"]),
            )
        return state["service"]

    def ok(data=None, status=200):
        return jsonify({"code": 0, "message": "success", "data": data}), status

    def fail(message, status=400):
        return jsonify({"code": status, "message": message, "data": None}), status

    def require_auth(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user"):
                return fail("请先登录", 401)
            return view(*args, **kwargs)
        return wrapped

    def int_arg(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(request.args.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
        return value

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    @app.get("/api/health")
    def health():
        ready = all(Path(app.config[key]).exists() for key in ("SCORE_PATH", "CATALOG_PATH", "NEIGHBORS_PATH"))
        return ok({"status": "ok", "model_ready": ready})

    @app.get("/api/system")
    def system_info():
        return ok(service().system_info())

    @app.get("/api/auth/demo-accounts")
    def demo_accounts():
        return ok([
            {"username": username, "display_name": display_name, "password": password}
            for username, _, display_name, password in DEMO_ACCOUNTS
        ])

    @app.post("/api/auth/login")
    def login():
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
        session.clear()
        return ok({"logged_out": True})

    @app.get("/api/auth/status")
    def auth_status():
        return ok({"authenticated": bool(session.get("user")), "user": session.get("user")})

    @app.get("/api/auth/me")
    @require_auth
    def me():
        return ok(session["user"])

    @app.get("/api/recommendations")
    @require_auth
    def recommendations():
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
        return ok({"item": service().product(item_id), "similar": service().similar(item_id, 6)})

    @app.get("/api/items/<item_id>/similar")
    @require_auth
    def similar(item_id):
        try:
            limit = int_arg("limit", 6, 1, 12)
        except ValueError as exc:
            return fail(str(exc))
        exclude = set(filter(None, request.args.get("exclude", "").split(",")))
        return ok({"item_id": item_id, "items": service().similar(item_id, limit, exclude)})

    @app.post("/api/events/click")
    @require_auth
    def click_event():
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
        return fail("接口不存在", 404)

    @app.errorhandler(405)
    def method_not_allowed(_):
        return fail("请求方法不允许", 405)

    return app


app = create_app()
