"""补货召回通道（Replenishment Route）。

课程主线四路召回（popular/itemcf/item2vec/uservec）均面向"新品发现"，
在 candidates() 中一律排除用户已购商品。但本数据为 B2B 批发场景，
实测约 18% 的未来购买是对历史商品的重复采购（补货），且补货信号强度
远高于共购结构信号：纯补货通道在含复购口径下 P@10≈0.27，是共购通道
(≈0.085) 的三倍余（见 outputs/baseline 下双轨评估）。

本模块提供第五路"补货召回"：对用户已购过的商品，按 RFM-lite 打分
（购买频次 × 近因衰减），分数越高表示越可能再次采购。该通道【不排除】
已购商品——这正是它与其余四路的本质区别，也是双轨评估的意义所在。

设计约束：
- 仅使用 history 期（< cfg.history_end）信息，无未来泄漏；
- 与 baseline 其余模块同风格：numpy 向量化、可被 engine 直接调用；
- 退化为纯频次排序时（half_life_days→∞）等价于经典 RFM 的 F 维度。
"""
import numpy as np
import pandas as pd


# Online Retail 中的非产品代码（运费/邮费/手工调整/银行手续费等财务调整项）。
# 实测在 history 期共 802 行（占 0.4%）：POST(570) M(157) C2(69) BANK CHARGES(4) PADS(2)。
# 课程 baseline 的三级清洗漏斗未剔除它们（见 data.clean_data），因此评估用的
# truth 口径保持与官方基线一致、不做修改；仅在补货通道【内部】跳过这些代码，
# 避免 Web 演示出现"C2（运费）建议补货"的业务错误。
# 注意：这是单向过滤（只减候选不减 truth），故只会低估补货通道指标，属保守处理。
NON_PRODUCT_CODES = frozenset({
    'B', 'C2', 'DOT', 'M', 'S', 'D', 'D1', 'D2', 'CR', 'ADJUST',
    'PADS', 'POST', 'BANK CHARGES', 'AMAZONFEE', 'gift', 'm', 'SALVAGE',
})


class ReplenishmentRoute:
    """按用户构建"已购商品 → 补货分数"的召回通道。

    参数
    ----
    history : DataFrame
        清洗后、且已按 cfg.history_end 截断的历史交易帧，
        须含 user_id / item_id / transaction_time 三列。
    item_index : dict[str, int]
        商品 ID → 全局位置下标（与 engine.pos 一致），保证打分向量
        可直接与其余四路在同一商品坐标系下融合。
    n_items : int
        商品总数，用于分配打分向量长度。
    half_life_days : float
        近因衰减半衰期（天）。距上次购买每过半衰期，分数权重减半。
    min_freq : int
        进入补货候选的最小历史购买次数（订单去重）。消融实测 min_freq=1
        优于 2/3（all 口径 P@10 0.171 vs 0.130 vs 0.081，覆盖用户 3145 vs
        1566 vs 800），即"只买过一次"的商品同样是有效补货候选；"批发补货
        须至少复购过一次"的先验被数据否定，故默认取 1。
    """

    def __init__(self, history, item_index, n_items, half_life_days=60.0, min_freq=1):
        self.item_index = item_index
        self.n_items = n_items
        self.half_life_days = float(half_life_days)
        self.min_freq = int(min_freq)
        # 以 history 期最晚时间为"当前时刻"，避免用到任何未来时间戳。
        self.as_of = pd.Timestamp(history.transaction_time.max())
        self.scores = self._build(history)

    def _build(self, history):
        """构建 user_id → 长度 n_items 的补货分数向量（稀疏字典）。"""
        frame = history[['user_id', 'item_id', 'invoice_no', 'transaction_time']].copy()
        frame = frame[frame.item_id.isin(self.item_index)]
        # 剔除非产品代码（运费/邮费/调整项）：仅收窄本通道候选，不改 truth 口径（见模块头注释）
        frame = frame[~frame.item_id.isin(NON_PRODUCT_CODES)]
        scores = {}
        # 按 (用户, 商品) 聚合：购买订单数 = 频次，最晚购买时间 = 近因基准。
        grouped = frame.groupby(['user_id', 'item_id'], sort=False)
        agg = grouped.agg(freq=('invoice_no', 'nunique'),
                          last=('transaction_time', 'max')).reset_index()
        for user, block in agg.groupby('user_id', sort=False):
            vec = np.zeros(self.n_items, dtype=np.float32)
            for item, freq, last in zip(block.item_id, block.freq, block.last):
                if freq < self.min_freq:
                    continue
                pos = self.item_index.get(item)
                if pos is None:
                    continue
                gap_days = (self.as_of - last).days
                recency = 0.5 ** (gap_days / self.half_life_days)
                vec[pos] = freq * recency
            if vec.any():
                scores[str(user)] = vec
        return scores

    def score(self, user_id):
        """返回该用户的补货分数向量；无补货候选时返回全零（融合时自然垫底）。"""
        return self.scores.get(str(user_id), np.zeros(self.n_items, dtype=np.float32))

    def recommend(self, user_id, k):
        """返回该用户补货 Top-k 商品位置下标（按分数降序，ID 字典序 tie-break）。

        下标坐标系与 engine.ids 一致，调用方用 engine.ids[i] 还原商品 ID。
        """
        vec = self.score(user_id)
        eligible = np.flatnonzero(vec > 0)
        if len(eligible) == 0:
            return []
        order = np.lexsort((eligible, -vec[eligible]))
        return eligible[order[:k]].tolist()
