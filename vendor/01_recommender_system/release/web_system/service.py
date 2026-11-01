"""Self-contained serving adapter for Oblivionis v2.

Only frozen recommendation scores, product metadata and a stripped item-neighbor
index are loaded.  No raw transactions or pickle model are shipped in the web
runtime.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from threading import Lock
import time
import uuid

from werkzeug.security import generate_password_hash

from web_system.database import connect, record_feedback


MODEL_NAME = "Oblivionis"
IMPLEMENTATION = "v2-r1-head48"
MODEL_VERSION = f"{MODEL_NAME}({IMPLEMENTATION})"
DEMO_ACCOUNTS = (
    ("northstar", "14911.0", "北辰采购", "demo123"),
    ("paperplane", "12748.0", "纸飞机商店", "demo123"),
    ("greengate", "14156.0", "绿门生活馆", "demo123"),
)


class TTLCache:
    def __init__(self, ttl_seconds: int = 90):
        self.ttl_seconds = ttl_seconds
        self._data: dict[tuple, tuple[float, object]] = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            expires_at, payload = value
            if expires_at <= time.time():
                self._data.pop(key, None)
                return None
            return payload

    def set(self, key, payload):
        with self._lock:
            self._data[key] = (time.time() + self.ttl_seconds, payload)


class RecommendationService:
    def __init__(self, score_path: Path, catalog_path: Path, neighbors_path: Path, db_path: Path):
        self.write_lock = Lock()
        self.cache = TTLCache()
        self.rankings = self._load_rankings(score_path)
        self.catalog = self._load_catalog(catalog_path)
        self.neighbors = json.loads(neighbors_path.read_text(encoding="utf-8"))
        self.global_ranking = self._global_ranking()
        self.connection = connect(db_path)
        self._seed_database()

    @staticmethod
    def _load_rankings(path: Path) -> dict[str, list[tuple[str, float]]]:
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                grouped[str(row["user_id"])].append((str(row["item_id"]), float(row["score"])))
        rankings = {}
        for user_id, rows in grouped.items():
            rows.sort(key=lambda value: (-value[1], value[0]))
            denominator = max(1, len(rows) - 1)
            rankings[user_id] = [
                (item_id, 1.0 - index / denominator)
                for index, (item_id, _) in enumerate(rows)
            ]
        return rankings

    @staticmethod
    def _load_catalog(path: Path) -> dict[str, dict]:
        catalog = {}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                item_id = str(row["item_id"])
                catalog[item_id] = {
                    "item_id": item_id,
                    "name": row.get("name") or row.get("description") or f"商品 {item_id}",
                    "description": row.get("description") or "精选商品",
                    "category": row.get("category") or "精选好物",
                }
        return catalog

    def _global_ranking(self) -> list[tuple[str, float]]:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for rows in self.rankings.values():
            for item_id, score in rows:
                totals[item_id] += score
                counts[item_id] += 1
        return sorted(
            ((item_id, totals[item_id] / counts[item_id]) for item_id in totals),
            key=lambda value: (-value[1], value[0]),
        )

    def _seed_database(self) -> None:
        with self.write_lock:
            for ranking_user_id in self.rankings:
                self.connection.execute(
                    "INSERT OR IGNORE INTO users(user_id,country) VALUES(?,?)",
                    (f"{ranking_user_id}.0", "United Kingdom"),
                )
            for item_id in {item for rows in self.rankings.values() for item, _ in rows}:
                product = self.product(item_id)
                self.connection.execute(
                    "INSERT OR IGNORE INTO products(item_id,description,chinese_name,category) VALUES(?,?,?,?)",
                    (item_id, product["description"], product["name"], product["category"]),
                )
            for username, user_id, display_name, password in DEMO_ACCOUNTS:
                self.connection.execute(
                    "INSERT OR IGNORE INTO auth_accounts(username,user_id,display_name,password_hash) VALUES(?,?,?,?)",
                    (username, user_id, display_name, generate_password_hash(password)),
                )
            self.connection.commit()

    @staticmethod
    def _ranking_user_id(user_id: str) -> str:
        return str(user_id).removesuffix(".0")

    def account(self, username: str):
        return self.connection.execute(
            "SELECT username,user_id,display_name,password_hash FROM auth_accounts WHERE username=?",
            (username,),
        ).fetchone()

    def product(self, item_id: str) -> dict:
        item_id = str(item_id)
        return dict(self.catalog.get(item_id, {
            "item_id": item_id,
            "name": f"精选商品 {item_id}",
            "description": "该商品暂缺目录描述",
            "category": "精选好物",
        }))

    def system_info(self) -> dict:
        return {
            "model_name": MODEL_NAME,
            "implementation": IMPLEMENTATION,
            "model_version": MODEL_VERSION,
            "online_score": 0.8060,
            "ndcg_at_10": 0.8992,
            "recall_at_10": 0.4728,
            "gauc": 0.8793,
            "users": len(self.rankings),
            "candidates": sum(map(len, self.rankings.values())),
            "serving_mode": "冻结 v2 排序 + 脱敏相似商品索引",
        }

    def recommendations(self, user_id: str, cursor: int, limit: int, exclude: set[str]) -> dict:
        ranking_id = self._ranking_user_id(user_id)
        ranking = self.cache.get(("ranking", ranking_id))
        if ranking is None:
            ranking = self.rankings.get(ranking_id, self.global_ranking)
            self.cache.set(("ranking", ranking_id), ranking)
        items = []
        position = cursor
        while position < len(ranking) and len(items) < limit:
            item_id, score = ranking[position]
            position += 1
            if item_id in exclude or item_id not in self.catalog:
                continue
            items.append({
                **self.product(item_id),
                "score": round(float(score), 6),
                "source_type": "oblivionis_v2",
                "reason_code": "champion_personalized_rank",
                "reason": f"Oblivionis(R1) 用户内融合排名第 {position} 位",
            })
        session_id = str(uuid.uuid4())
        with self.write_lock:
            self.connection.execute(
                "INSERT INTO recommendation_sessions(session_id,user_id,model_version,request_k) VALUES(?,?,?,?)",
                (session_id, user_id, MODEL_VERSION, limit),
            )
            for rank, item in enumerate(items, 1):
                self.connection.execute(
                    "INSERT INTO recommendation_items(session_id,item_id,rank_position,final_score,source_type) VALUES(?,?,?,?,?)",
                    (session_id, item["item_id"], rank, item["score"], item["source_type"]),
                )
            self.connection.commit()
        for item in items:
            item["session_id"] = session_id
        return {
            "items": items,
            "session_id": session_id,
            "next_cursor": position if position < len(ranking) else None,
            "has_more": position < len(ranking),
            "model_version": MODEL_VERSION,
        }

    def similar(self, item_id: str, limit: int, exclude: set[str] | None = None) -> list[dict]:
        excluded = set(map(str, exclude or set())) | {str(item_id)}
        rows = []
        for neighbor_id, score in self.neighbors.get(str(item_id), []):
            if neighbor_id in excluded or neighbor_id not in self.catalog:
                continue
            rows.append({
                **self.product(neighbor_id),
                "score": round(float(score), 6),
                "source_type": "itemcf",
                "reason_code": "similar_to_clicked",
                "reason": "与你刚刚查看的商品经常被一起购买",
            })
            if len(rows) >= limit:
                return rows
        for fallback_id, score in self.global_ranking:
            if (
                fallback_id in excluded
                or fallback_id not in self.catalog
                or any(row["item_id"] == fallback_id for row in rows)
            ):
                continue
            rows.append({
                **self.product(fallback_id),
                "score": round(float(score), 6),
                "source_type": "champion_fallback",
                "reason_code": "popular_fallback",
                "reason": "候选集中的高质量商品",
            })
            if len(rows) >= limit:
                break
        return rows

    def feedback(self, user_id: str, payload: dict) -> None:
        event = {
            "session_id": payload.get("session_id"),
            "user_id": user_id,
            "item_id": str(payload["item_id"]),
            "feedback_type": payload.get("feedback_type", "click"),
            "idempotency_key": payload.get("idempotency_key", str(uuid.uuid4())),
            "client_time": payload.get("client_time"),
        }
        with self.write_lock:
            record_feedback(self.connection, event)
