"""模块一的 sqlite 持久化：建库建表 + 写入反馈事件。

本仓库这份是**只读副本，为了讲解补充了中文注释**（逻辑与上游原项目一致，AST 已对比过）。
模块三（阶段3）不直接调用这里的任何函数；模块一的 web_system/service.py 会用它建库、写反馈。
阅读时要能说清：六张表各存什么、`connect` 和 `record_feedback` 各接收什么参数。
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


# 六张表的建表语句（IF NOT EXISTS：每次启动执行都不会报错）
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users(                     -- 用户：user_id 形如 '14911.0'
    user_id TEXT PRIMARY KEY,
    country TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS auth_accounts(             -- 登录账号：用户名 + 密码哈希 + 对应 user_id
    username TEXT PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS products(                  -- 商品基础信息（从 catalog.csv 抄一份）
    item_id TEXT PRIMARY KEY,
    description TEXT,
    chinese_name TEXT,
    category TEXT
);
CREATE TABLE IF NOT EXISTS recommendation_sessions(   -- 每次推荐请求一行：谁、用的哪个模型、要几条
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    request_k INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS recommendation_items(      -- 每次推荐返回了什么：第几名、多少分、来源
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    rank_position INTEGER NOT NULL,
    final_score REAL NOT NULL,
    source_type TEXT,
    PRIMARY KEY(session_id,item_id),
    UNIQUE(session_id,rank_position),
    FOREIGN KEY(session_id) REFERENCES recommendation_sessions(session_id),
    FOREIGN KEY(item_id) REFERENCES products(item_id)
);
CREATE TABLE IF NOT EXISTS feedback_events(           -- 点击/喜欢/不喜欢；idempotency_key 保证不重复记
    feedback_id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK(feedback_type IN('click','like','dislike')),
    idempotency_key TEXT UNIQUE NOT NULL,
    client_time DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES recommendation_sessions(session_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(item_id) REFERENCES products(item_id)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    """连上 sqlite（没有就建库）并确保六张表存在。

    参数：
        path (Path): sqlite 文件路径。目录不存在会自动创建
            （模块三把它指到自己的 `runtime/module_one/serving.sqlite3`）。

    返回：
        sqlite3.Connection: 已打开、已建表、已开外键约束的连接。
        注意 `check_same_thread=False`：因为 Flask 是多线程的，连接要跨线程复用。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    return connection


def record_feedback(connection: sqlite3.Connection, event: dict) -> None:
    """把一条反馈事件写进 `feedback_events` 表。

    参数：
        connection (sqlite3.Connection): `connect()` 返回的连接。
        event (dict): 一条反馈，字段：
            - feedback_id (str, 可选): 不传就生成 UUID；
            - session_id (str | None): 关联的推荐会话；
            - user_id (str, 必填): 谁点的；
            - item_id (str, 必填): 点的哪个商品；
            - feedback_type (str, 必填): click / like / dislike；
            - idempotency_key (str, 必填): 幂等键，重复提交不会写两条；
            - client_time (str | None): 客户端时间。

    返回：
        None。副作用：写库并 commit。

    说明：
        - 用 `INSERT OR IGNORE`，所以同一个 idempotency_key 再来一次会被忽略（幂等）；
        - 模块三的点赞/点踩**不走这个函数**，只用自己会话里的偏好。
    """
    connection.execute(
        """INSERT OR IGNORE INTO feedback_events(
               feedback_id,session_id,user_id,item_id,feedback_type,idempotency_key,client_time
           ) VALUES(?,?,?,?,?,?,?)""",
        (
            event.get("feedback_id", str(uuid.uuid4())),
            event.get("session_id"),
            event["user_id"],
            event["item_id"],
            event["feedback_type"],
            event["idempotency_key"],
            event.get("client_time"),
        ),
    )
    connection.commit()
