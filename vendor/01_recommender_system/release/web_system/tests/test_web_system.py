from pathlib import Path

import pytest

from web_system.app import create_app
from web_system.service import RecommendationService


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def system(tmp_path):
    service = RecommendationService(
        ROOT / "expected" / "submission_v2_R1_head48.csv",
        ROOT / "web_system" / "data" / "catalog.csv",
        ROOT / "web_system" / "data" / "item_neighbors.json",
        tmp_path / "test.sqlite3",
    )
    app = create_app({"TESTING": True, "SECRET_KEY": "test"}, service)
    yield app.test_client(), service
    service.connection.close()


def login(client):
    response = client.post("/api/auth/login", json={"username": "northstar", "password": "demo123"})
    assert response.status_code == 200


def test_health_and_system(system):
    client, _ = system
    health = client.get("/api/health").get_json()["data"]
    assert health == {"status": "ok", "model_ready": True}
    system = client.get("/api/system").get_json()["data"]
    assert system["online_score"] == 0.806
    assert system["users"] == 736
    assert system["candidates"] == 58205


def test_login_recommend_click_and_feedback(system):
    client, service = system
    login(client)
    feed = client.get("/api/recommendations?cursor=0&limit=12")
    assert feed.status_code == 200
    payload = feed.get_json()["data"]
    assert len(payload["items"]) == 12
    item_id = payload["items"][0]["item_id"]
    click = client.post("/api/events/click", json={
        "item_id": item_id,
        "session_id": payload["session_id"],
        "visible_item_ids": [row["item_id"] for row in payload["items"]],
        "idempotency_key": "test-click",
    })
    assert click.status_code == 200
    assert click.get_json()["data"]["event_recorded"] is True
    like = client.post("/api/feedback", json={
        "item_id": item_id,
        "session_id": payload["session_id"],
        "feedback_type": "like",
        "idempotency_key": "test-like",
    })
    assert like.status_code == 200
    assert like.get_json()["data"] == {"recorded": True, "feedback_type": "like"}
    count = service.connection.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
    assert count == 2


def test_three_demo_accounts_are_personalized(system):
    client, _ = system
    feeds = []
    for username in ("northstar", "paperplane", "greengate"):
        response = client.post("/api/auth/login", json={"username": username, "password": "demo123"})
        assert response.status_code == 200
        items = client.get("/api/recommendations?cursor=0&limit=5").get_json()["data"]["items"]
        feeds.append(tuple(row["item_id"] for row in items))
    assert len(set(feeds)) == 3


def test_recommendation_pages_only_use_catalog_items(system):
    client, service = system
    for username in ("northstar", "paperplane", "greengate"):
        response = client.post("/api/auth/login", json={"username": username, "password": "demo123"})
        assert response.status_code == 200
        cursor = 0
        for _ in range(2):
            payload = client.get(f"/api/recommendations?cursor={cursor}&limit=12").get_json()["data"]
            assert len(payload["items"]) == 12
            assert all(row["item_id"] in service.catalog for row in payload["items"])
            assert all(row["description"] != "该商品暂缺目录描述" for row in payload["items"])
            cursor = payload["next_cursor"]


def test_similar_items_can_be_followed_multiple_levels(system):
    client, service = system
    login(client)
    feed = client.get("/api/recommendations?cursor=0&limit=1").get_json()["data"]
    first_id = feed["items"][0]["item_id"]
    level_one = client.get(f"/api/items/{first_id}/similar?limit=6").get_json()["data"]["items"]
    assert level_one
    second_id = level_one[0]["item_id"]
    level_two = client.get(f"/api/items/{second_id}/similar?limit=6").get_json()["data"]["items"]
    assert level_two
    assert all(row["item_id"] in service.catalog for row in [*level_one, *level_two])
