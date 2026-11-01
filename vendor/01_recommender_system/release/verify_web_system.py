"""Run a dependency-light end-to-end check of the packaged Flask system."""
from __future__ import annotations

from pathlib import Path
import tempfile

from web_system.app import create_app
from web_system.service import RecommendationService


ROOT = Path(__file__).resolve().parent


def main() -> None:
    forbidden = {"engine.pkl", "train.csv"}
    shipped_names = {path.name.lower() for path in (ROOT / "web_system").rglob("*") if path.is_file()}
    assert not (forbidden & shipped_names), "网页系统不得携带原始交易数据或含历史记录的引擎"

    with tempfile.TemporaryDirectory(prefix="oblivionis-web-") as temp_dir:
        service = RecommendationService(
            ROOT / "expected" / "submission_v2_R1_head48.csv",
            ROOT / "web_system" / "data" / "catalog.csv",
            ROOT / "web_system" / "data" / "item_neighbors.json",
            Path(temp_dir) / "test.sqlite3",
        )
        app = create_app({"TESTING": True, "SECRET_KEY": "verify"}, service)
        client = app.test_client()

        try:
            health = client.get("/api/health")
            assert health.status_code == 200 and health.get_json()["data"]["model_ready"]
            system = client.get("/api/system").get_json()["data"]
            assert system["online_score"] == 0.806 and system["users"] == 736

            feeds = {}
            for username in ("northstar", "paperplane", "greengate"):
                login = client.post("/api/auth/login", json={"username": username, "password": "demo123"})
                assert login.status_code == 200
                account_feed = client.get("/api/recommendations?cursor=0&limit=5").get_json()["data"]
                assert all(row["item_id"] in service.catalog for row in account_feed["items"])
                feeds[username] = tuple(row["item_id"] for row in account_feed["items"])
            assert len(set(feeds.values())) == 3, "三个演示账号应得到不同推荐"

            client.post("/api/auth/login", json={"username": "northstar", "password": "demo123"})
            feed = client.get("/api/recommendations?cursor=0&limit=12")
            payload = feed.get_json()["data"]
            assert feed.status_code == 200 and len(payload["items"]) == 12
            assert all(row["description"] != "该商品暂缺目录描述" for row in payload["items"])
            next_payload = client.get(
                f"/api/recommendations?cursor={payload['next_cursor']}&limit=12"
            ).get_json()["data"]
            assert {row["item_id"] for row in payload["items"]}.isdisjoint(
                row["item_id"] for row in next_payload["items"]
            )

            first = payload["items"][0]
            detail = client.get(f"/api/items/{first['item_id']}")
            detail_item = detail.get_json()["data"]["item"]
            assert detail.status_code == 200 and detail_item["item_id"] == first["item_id"]
            level_one = client.get(f"/api/items/{first['item_id']}/similar?limit=6").get_json()["data"]["items"]
            assert level_one
            level_two = client.get(
                f"/api/items/{level_one[0]['item_id']}/similar?limit=6"
            ).get_json()["data"]["items"]
            assert level_two
            click = client.post("/api/events/click", json={
                "item_id": first["item_id"],
                "session_id": payload["session_id"],
                "visible_item_ids": [row["item_id"] for row in payload["items"]],
                "idempotency_key": "verify-click",
            })
            assert click.status_code == 200 and click.get_json()["data"]["event_recorded"]

            like = client.post("/api/feedback", json={
                "item_id": first["item_id"],
                "session_id": payload["session_id"],
                "feedback_type": "like",
                "idempotency_key": "verify-like",
            })
            assert like.status_code == 200 and like.get_json()["data"]["recorded"]
            events = service.connection.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
            assert events == 2
        finally:
            service.connection.close()

    print("PASS: clean catalog cards, rotating pages, multi-level detail, click insertion, like persistence")


if __name__ == "__main__":
    main()
