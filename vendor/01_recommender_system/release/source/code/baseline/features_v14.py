"""v14 扩展季节/年度同比特征（在 v11 的 10 个静态特征之上叠加）。

动机
----
v11 的 10 个特征把 2009-12 视为整体"圣诞月"，只回答了：

    这个 (user, item) 去年 12 月买过没有 / 数量多少？

但 v11 OOF 0.884 → 在线 0.7792 仍然有 0.10 的差距，说明单凭"去年 12 月"已
经不够。v14 沿三个轴扩展：

1. **item 维度加入价格与篮子信号**：
   - `item_dec09_log_avg_price`：2009-12 该商品平均成交价。
     圣诞礼物/装饰品单价通常高于该商品日常均价，这是一个"是否礼品"信号。
   - `item_dec09_avg_basket_quantity`：2009-12 该商品每单平均件数。
     整篮采购是 B2B 圣诞补货的典型模式，与个人购买不同。
2. **user 维度加入多样性 + 篮子信号**：
   - `user_dec09_log_item_diversity`：2009-12 该用户买过的不同商品数。
     圣诞采购客单 SKU 多（装饰+礼物+包装纸），是一个"用户是否圣诞型"信号。
   - `user_dec09_avg_basket_quantity`：2009-12 该用户每单平均件数。
3. **pair 维度补足"价格与件数"**（v11 只有 orders 和 quantity 的对数）：
   - `pair_dec09_log_avg_price` / `pair_dec09_avg_quantity`。
4. **item 描述关键词扩展**：v11 只有 CHRISTMAS/XMAS/SANTA/HALLOWEEN/WINTER/
   STOCKING/SEASON；v14 加上 GIFT/WRAP/TAG/CARD/RIBBON/BOW/DECOR/STAR/TREE/
   HEART/CANDLE/LANTERN/SNOW/FROST 等等，覆盖描述里"包装类""装饰类"等
   没被 v11 抓到的"广义节日"信号。
5. **user × item 双活跃交叉**：
   - `pair_dec09_user_x_item_active`：用户 2009-12 活跃 AND 商品 2009-12
     有交易。这一列不是新信号，但把两个 0/1 特征相乘，让树模型能直接
     学到"双方都活跃"的高阶交互。

防泄漏论证
----------
- 2009-12 窗口：[2009-12-01, 2010-01-01)
- 2010-01/02 对照窗：[2010-01-01, 2010-03-01)
- 全部窗口严格截止于 2010-03-01 之前，早于最早回放月 2010-06-01，零泄漏。
- 跟 v11 共用同一条 seasonal_state 持久化路径（v14_state.pkl），训练/提交
  两侧特征完全一致。

用法
----
    from baseline.features_v14 import build_v14_state, append_v14_frame

    state = build_v14_state(clean)
    frame = append_v14_frame(frame, state)  # frame 含 user_id, item_id
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRA_FEATURES_V14 = [
    # —— item 维度 ——
    "item_dec09_log_avg_price",
    "item_dec09_avg_basket_quantity",
    "item_gift_keyword",
    # —— user 维度 ——
    "user_dec09_log_item_diversity",
    "user_dec09_avg_basket_quantity",
    "user_dec09_log_avg_price",
    # —— pair 维度 ——
    "pair_dec09_log_avg_price",
    "pair_dec09_avg_quantity",
    # —— 双活跃交叉 ——
    "pair_dec09_user_x_item_active",
]

# 广义节日/礼品/装饰关键词。v11 的 CHRISTMAS/XMAS 等已被这条更长的列表覆盖。
# 注意：单词边界要小心——'WOOD' 不会被 'STAR' 命中，所以直接 substring 即可。
GIFT_KEYWORDS = (
    # 礼品/包装
    "GIFT", "WRAP", "WRAPPING", "TAG", "RIBBON", "BOW", "CARD",
    "BAG", "BOX", "PACK", "TISSUE", "PRESENT",
    # 装饰/圣诞
    "DECOR", "DECORATION", "ORNAMENT", "WREATH", "GARLAND",
    "STAR", "TREE", "CANDLE", "LANTERN", "TEALIGHT",
    "SNOW", "FROST", "ICE", "WINTER",
    "HEART", "LOVE", "VALENTINE",  # Q1 节庆
    "BIRTHDAY", "PARTY",  # 礼品的非圣诞场景
    "EASTER", "EGG", "BUNNY",  # 春季节庆
    # 直接节日
    "CHRISTMAS", "XMAS", "SANTA", "NOEL", "REINDEER",
    "STOCKING", "MISTLETOE", "HOLLY", "IVY", "POINSETTIA",
    "PUMPKIN", "HALLOWEEN",  # 万圣
)


def _gift_keyword_flag(description) -> int:
    if not isinstance(description, str):
        return 0
    upper = description.upper()
    return int(any(keyword in upper for keyword in GIFT_KEYWORDS))


def _pair_key(users: pd.Series, items: pd.Series) -> pd.Series:
    return users.astype(str).str.cat(items.astype(str), sep="\x1f")


def build_v14_state(clean: pd.DataFrame) -> dict:
    """一次性计算 v14 全部静态特征，存为可 pickle 的 state。

    依赖列：user_id / item_id / invoice_no / transaction_time / Quantity /
    UnitPrice / description。
    """
    required = {
        "user_id", "item_id", "invoice_no",
        "transaction_time", "Quantity", "UnitPrice", "description",
    }
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"clean 缺少列：{sorted(missing)}")

    time = clean.transaction_time
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("transaction_time 必须为 datetime64")

    dec09 = clean.loc[
        (time >= pd.Timestamp("2009-12-01"))
        & (time < pd.Timestamp("2010-01-01"))
    ].copy()
    del time

    # —— item 维度：均价、basket、关键词 ——
    # 单价过滤：UnitPrice > 0，Quantity > 0，避免退单把均价拉成 0 或负
    dec_pos = dec09.loc[(dec09.UnitPrice > 0) & (dec09.Quantity > 0)].copy()
    item_dec_price = dec_pos.groupby("item_id").UnitPrice.mean()
    item_dec_qty = dec_pos.groupby(["item_id", "invoice_no"]).Quantity.sum()
    item_dec_avg_basket = item_dec_qty.groupby("item_id").mean()

    descriptions = (
        clean[["item_id", "description"]]
        .dropna(subset=["description"])
        .groupby("item_id").description.first()
    )
    item_gift = {
        str(item): _gift_keyword_flag(desc)
        for item, desc in descriptions.items()
    }

    # —— user 维度：多样性、basket、均价 ——
    user_dec_items = dec09.groupby("user_id").item_id.nunique()
    user_dec_qty = dec09.groupby(["user_id", "invoice_no"]).Quantity.sum()
    user_dec_avg_basket = user_dec_qty.groupby("user_id").mean()
    user_dec_price = dec_pos.groupby("user_id").UnitPrice.mean()

    # —— pair 维度：均价、件数（按单求和后的均值）——
    pair_dec_price = dec_pos.groupby(["user_id", "item_id"]).UnitPrice.mean()
    pair_dec_qty = dec09.groupby(["user_id", "item_id"]).Quantity.sum().clip(lower=0)

    pair_price_key = _pair_key(
        pd.Series(pair_dec_price.index.get_level_values(0).astype(str)),
        pd.Series(pair_dec_price.index.get_level_values(1).astype(str)),
    )
    pair_qty_key = _pair_key(
        pd.Series(pair_dec_qty.index.get_level_values(0).astype(str)),
        pd.Series(pair_dec_qty.index.get_level_values(1).astype(str)),
    )

    # —— user × item 双活跃交叉（提前算好 pair key dict）——
    user_dec_active_set = set(dec09.user_id.astype(str).unique())
    item_dec_active_set = set(dec09.item_id.astype(str).unique())

    state = {
        "item_dec09_avg_price": item_dec_price.astype(float).to_dict(),
        "item_dec09_avg_basket": item_dec_avg_basket.astype(float).to_dict(),
        "item_gift_keyword": item_gift,
        "user_dec09_item_diversity": user_dec_items.astype(float).to_dict(),
        "user_dec09_avg_basket": user_dec_avg_basket.astype(float).to_dict(),
        "user_dec09_avg_price": user_dec_price.astype(float).to_dict(),
        "pair_dec09_avg_price": {
            key: float(value) for key, value in zip(pair_price_key, pair_dec_price.astype(float))
        },
        "pair_dec09_quantity": {
            key: float(value) for key, value in zip(pair_qty_key, pair_dec_qty.astype(float))
        },
        "user_dec09_active_set": user_dec_active_set,
        "item_dec09_active_set": item_dec_active_set,
    }
    return state


def _feature_values(users: pd.Series, items: pd.Series, state: dict) -> dict:
    users = users.astype(str).reset_index(drop=True)
    items = items.astype(str).reset_index(drop=True)
    pair_keys = users.str.cat(items, sep="\x1f")

    item_price = items.map(state["item_dec09_avg_price"]).fillna(0.0).to_numpy(float)
    item_basket = items.map(state["item_dec09_avg_basket"]).fillna(0.0).to_numpy(float)
    item_gift = items.map(state["item_gift_keyword"]).fillna(0).to_numpy(float)

    user_diversity = users.map(state["user_dec09_item_diversity"]).fillna(0.0).to_numpy(float)
    user_basket = users.map(state["user_dec09_avg_basket"]).fillna(0.0).to_numpy(float)
    user_price = users.map(state["user_dec09_avg_price"]).fillna(0.0).to_numpy(float)

    pair_price = pair_keys.map(state["pair_dec09_avg_price"]).fillna(0.0).to_numpy(float)
    pair_qty = pair_keys.map(state["pair_dec09_quantity"]).fillna(0.0).to_numpy(float)

    # 双活跃交叉：用户活跃 AND 商品活跃（都是 0/1）
    user_active = np.fromiter(
        (1.0 if u in state["user_dec09_active_set"] else 0.0 for u in users),
        dtype=float, count=len(users),
    )
    item_active = np.fromiter(
        (1.0 if i in state["item_dec09_active_set"] else 0.0 for i in items),
        dtype=float, count=len(items),
    )
    both_active = user_active * item_active

    return {
        "item_dec09_log_avg_price": np.log1p(item_price),
        "item_dec09_avg_basket_quantity": item_basket,
        "item_gift_keyword": item_gift,
        "user_dec09_log_item_diversity": np.log1p(user_diversity),
        "user_dec09_avg_basket_quantity": user_basket,
        "user_dec09_log_avg_price": np.log1p(user_price),
        "pair_dec09_log_avg_price": np.log1p(pair_price),
        "pair_dec09_avg_quantity": pair_qty,
        "pair_dec09_user_x_item_active": both_active,
    }


def append_v14_frame(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    """给整表追加 v14 特征列（frame 需含 user_id / item_id），返回副本。"""
    if "user_id" not in frame.columns or "item_id" not in frame.columns:
        raise ValueError("frame 必须包含 user_id 与 item_id 两列")
    values = _feature_values(frame["user_id"], frame["item_id"], state)
    result = frame.copy()
    for name in EXTRA_FEATURES_V14:
        result[name] = values[name]
    return result


def add_v14_columns(base: pd.DataFrame, state: dict, uid, item_ids) -> pd.DataFrame:
    """提交期逐用户追加 v14 特征：base 是单用户特征表，item_ids 与之等长。"""
    length = len(base)
    if length != len(item_ids):
        raise ValueError("base 与 item_ids 长度不一致")
    values = _feature_values(
        pd.Series([str(uid)] * length), pd.Series(list(item_ids)), state
    )
    return pd.DataFrame(
        {name: values[name] for name in EXTRA_FEATURES_V14}, index=base.index
    )
