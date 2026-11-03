"""模块一的推荐服务实现：读冻结排序 + 商品表 + 相似索引，提供按用户的推荐、分页和反馈。

本仓库这份是**只读副本，为了讲解补充了中文注释**（逻辑与上游原项目一致，AST 已对比过）。
模块三（阶段3）只用到其中三样：
    1. `RecommendationService.__init__` 装载四个数据文件；
    2. `recommendations(user_id, cursor, limit, exclude)` 按用户返回排序候选（分页）；
    3. `account(username)`（登录时查账号）+ `product(item_id)`（补商品属性）。

关键点（答辩常问）：CSV 里的原始分数是负数（如 -95、-24），但 `_load_rankings` 会
**按名次归一化成 0~1**（第一名 1.0、最后一名 0.0），模块三拿到的就是这 0~1 的分。
阅读约定：每个函数都写了「参数 / 返回 / 怎么做」。
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


MODEL_NAME = "Oblivionis"                                  # 模型名
IMPLEMENTATION = "v2-r1-head48"                            # 具体实现版本
MODEL_VERSION = f"{MODEL_NAME}({IMPLEMENTATION})"          # 对外展示的版本串
DEMO_ACCOUNTS = (                                          # 发布页公开的三个演示账号（非私有密钥）
    ("northstar", "14911.0", "北辰采购", "demo123"),
    ("paperplane", "12748.0", "纸飞机商店", "demo123"),
    ("greengate", "14156.0", "绿门生活馆", "demo123"),
)


class TTLCache:
    """带过期时间的线程安全小缓存（给“用户排名”用，默认 90 秒）。"""

    def __init__(self, ttl_seconds: int = 90):
        """参数：ttl_seconds —— 每条缓存多少秒后过期。"""
        self.ttl_seconds = ttl_seconds
        self._data: dict[tuple, tuple[float, object]] = {}   # key -> (过期时间, 值)
        self._lock = Lock()

    def get(self, key):
        """取缓存。

        参数：
            key (tuple): 缓存键，本文件用 ('ranking', 用户ID)。
        返回：
            命中的值；不存在或已过期返回 None（过期会顺手删掉）。
        """
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
        """写缓存。

        参数：
            key (tuple): 缓存键；payload: 任意要缓存的对象（这里是该用户的排名列表）。
        返回：
            None。
        """
        with self._lock:
            self._data[key] = (time.time() + self.ttl_seconds, payload)


class RecommendationService:
    """推荐服务：构造时把四个文件读进内存，之后只读。"""

    def __init__(self, score_path: Path, catalog_path: Path, neighbors_path: Path, db_path: Path):
        """装载冻结排序、商品表、相似索引，并初始化 sqlite。

        参数：
            score_path (Path): 冻结排序 CSV（三列 user_id,item_id,score）。
            catalog_path (Path): 商品表 CSV（item_id,name,description,category）。
            neighbors_path (Path): 相似商品索引 JSON。
            db_path (Path): sqlite 文件路径；不存在会自动建库建表。

        副作用（构造完就有这些属性）：
            self.rankings        {用户ID: [(商品ID, 0~1 的分), ...]}（已按分数降序）
            self.catalog         {商品ID: {item_id,name,description,category}}
            self.neighbors       相似索引 {商品ID: [[邻居ID, 分], ...]}
            self.global_ranking  全站热门排序（冷启动兜底）
            self.connection      sqlite 连接
        """
        self.write_lock = Lock()                     # 写库时的互斥锁
        self.cache = TTLCache()                      # 用户排名缓存（90 秒）
        self.rankings = self._load_rankings(score_path)
        self.catalog = self._load_catalog(catalog_path)
        self.neighbors = json.loads(neighbors_path.read_text(encoding="utf-8"))
        self.global_ranking = self._global_ranking()
        self.connection = connect(db_path)
        self._seed_database()

    @staticmethod
    def _load_rankings(path: Path) -> dict[str, list[tuple[str, float]]]:
        """读冻结排序 CSV，并按“名次”归一化成 0~1 的分数。

        参数：
            path (Path): 冻结排序 CSV 路径（736 个用户，三列 user_id,item_id,score）。

        返回：
            dict: {用户ID: [(商品ID, 分), ...]}，每个用户的列表已按分数降序。
            分数不是 CSV 里的原始分（可能是 -95、-24），而是**名次分**：
                第 0 名 → 1.0；第 i 名 → 1.0 - i/(n-1)；最后一名 → 0.0。

        怎么做：
            1. 逐行读 CSV，按 user_id 分组收集 (item_id, 原始分)；
            2. 每组按 (-原始分, 商品ID) 排序；
            3. 用 `1.0 - index/(n-1)` 换成 0~1（n=该用户候选数；n=1 时分母取 1）。

        为什么这样：原始分是模型的 logit，量纲不统一；按名次归一化后，
        模块三可以直接把它当“上游推荐强度”用（第一名最强 = 1.0）。
        实测：用户 14911 有 344 条，相邻两名差 1/343 ≈ 0.0029。
        """
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                grouped[str(row["user_id"])].append((str(row["item_id"]), float(row["score"])))
        rankings = {}
        for user_id, rows in grouped.items():
            rows.sort(key=lambda value: (-value[1], value[0]))     # 分数降序，同分按商品ID
            denominator = max(1, len(rows) - 1)                    # n-1（只有一个候选时避免除 0）
            rankings[user_id] = [
                (item_id, 1.0 - index / denominator)               # 名次 → 0~1
                for index, (item_id, _) in enumerate(rows)
            ]
        return rankings

    @staticmethod
    def _load_catalog(path: Path) -> dict[str, dict]:
        """读商品表 CSV。

        参数：
            path (Path): catalog.csv（item_id,name,description,category）。
        返回：
            dict: {商品ID: {item_id, name, description, category}}；
            名字缺省用描述、再缺省用“商品 ID”；描述/品类缺省给兜底文案。
        """
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
        """算“全站热门”：把所有用户的名次分按商品求平均，作为冷启动兜底。

        参数：无（用 self.rankings）。
        返回：
            list[(商品ID, 平均分)]，按平均分降序。
        什么时候用：某个用户没有自己的排名（例如新用户）时，recommendations 会退回这个。
        """
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
        """第一次建库时写入基础数据（重复执行不会重复插）。

        参数：无。
        返回：None。
        写什么：
            users          每个有排名的用户写一条 "{user_id}.0"（所以模块三的账号是 14911.0）；
            products       排名里出现过的商品（id/描述/中文名/品类）；
            auth_accounts  三个演示账号 + 密码 hash（generate_password_hash）。
        """
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
        """把接口传来的 user_id 换成 CSV 里的写法。

        参数：
            user_id (str): 接口里的用户 ID，例如 '14911.0'。
        返回：
            str: 去掉 '.0' 后缀的 ID，例如 '14911'（CSV 里是这种写法）。
        """
        return str(user_id).removesuffix(".0")

    def account(self, username: str):
        """按用户名查账号（登录用）。

        参数：
            username (str): 已转小写的用户名。
        返回：
            sqlite 行 (username, user_id, display_name, password_hash)；不存在返回 None。
        """
        return self.connection.execute(
            "SELECT username,user_id,display_name,password_hash FROM auth_accounts WHERE username=?",
            (username,),
        ).fetchone()

    def product(self, item_id: str) -> dict:
        """按 ID 取商品；不在目录里时给兜底文案（不报错）。

        参数：
            item_id (str): 商品 ID。
        返回：
            dict: {item_id, name, description, category}。
        """
        item_id = str(item_id)
        return dict(self.catalog.get(item_id, {
            "item_id": item_id,
            "name": f"精选商品 {item_id}",
            "description": "该商品暂缺目录描述",
            "category": "精选好物",
        }))

    def system_info(self) -> dict:
        """返回版本与规模信息（`/api/system` 用，模块三不使用）。

        参数：无。
        返回：
            dict: 模型名、版本、线上指标（online_score/ndcg/recall/gauc）、用户数、候选数、服务模式。
        """
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
        """按用户返回推荐（分页），并把这次请求/结果写进数据库。

        参数：
            user_id (str): 登录用户，例如 '14911.0'（内部会去掉 .0 去查排名）。
            cursor (int): 从第几条开始（分页游标），0 表示从头。
            limit (int): 本次最多返回几条（接口限制 1~24）。
            exclude (set[str]): 要跳过的商品 ID（用户点过“不喜欢”的）。

        返回：
            dict: {
                items: [{item_id, name, description, category, score(0~1),
                         source_type:'oblivionis_v2', reason_code, reason, session_id}],
                session_id: 本次推荐会话 ID,
                next_cursor: 下次从哪条继续（到结尾时为 None）,
                has_more: 后面是否还有,
                model_version: 模型版本；
            }

        怎么做：
            1. 取该用户的排名（先查缓存；没有就用 self.rankings；再没有用全站热门）；
            2. 从 cursor 往后扫：跳过 exclude 里的、跳过不在商品目录里的，凑够 limit 条；
            3. 每条 = 商品属性 + score + 来源/理由（“Oblivionis(R1) 用户内融合排名第 N 位”）；
            4. 写一张 recommendation_sessions（本次请求）+ 若干 recommendation_items（返回了什么、第几名）；
            5. next_cursor = 扫到的位置，has_more = 位置是否还没到结尾。
        """
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
                continue                                   # 用户不想看的、或不在商品目录里的跳过
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
        """返回相似商品（模块三不使用，但要知道它是干什么的）。

        参数：
            item_id (str): 要找相似的商品。
            limit (int): 最多返回几个。
            exclude (set[str] | None): 要排除的商品 ID（含它自己）。
        返回：
            list[dict]: 商品属性 + score + source_type（itemcf 或 champion_fallback）+ reason。
        怎么做：先用 item_neighbors.json 的“一起买过”邻居；不够再用全站热门补足。
        """
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
        """记录点击/喜欢/不喜欢（模块三不使用）。

        参数：
            user_id (str): 用户 ID。
            payload (dict): 至少包含 item_id、feedback_type（click/like/dislike）；
                可选 session_id、idempotency_key（不传就生成）、client_time。
        返回：None。副作用：写 feedback_events 表（靠 idempotency_key 幂等）。
        """
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
