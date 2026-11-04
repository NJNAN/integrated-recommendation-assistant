"""按用户+会话隔离的短期上下文：约束可撤销，原始音视频一律不落库。"""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import time


@dataclass
class SessionContext:
    """一个会话的可恢复快照：约束、证据、偏好、最近展示商品与撤销栈。
    例如“背包预算300以内”写进 constraints，偏好记成 {商品ID: +1/-1}。
    """
    user_id: str
    session_id: str
    constraints: dict = field(default_factory=dict)
    slot_evidence: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)          # 商品ID -> +1/-1
    last_item_ids: list = field(default_factory=list)        # 最近一次展示，用于反馈校验
    last_intent: str = 'discover'
    target_item_ids: list = field(default_factory=list)
    top_k: int = 6
    history: list = field(default_factory=list, repr=False)  # 撤销栈
    updated_at: float = 0.0

    def public(self):
        """导出对外可见的会话字段，不泄露撤销栈和时间戳。
        例如接口返回只带 constraints、preferences 等字段。

        返回：
            dict: 去掉 history、updated_at 后的字段副本。
        """
        data = asdict(self)
        del data['history'], data['updated_at']
        return data

    def snapshot(self):
        """做一份深拷贝快照，供冲突或异常路径回滚。
        例如合并槽位失败前先 snapshot()，失败后 restore() 回到原状态。
        """
        return deepcopy(self.__dict__)

    def restore(self, snapshot):
        """用快照整体覆盖当前状态，约束、证据和撤销栈都会一并还原。
        例如回滚时调用 restore(snapshot)，等价于回到拍照那一刻。

        参数：
            snapshot (dict): snapshot() 返回的字段字典。
        """
        self.__dict__.clear()
        self.__dict__.update(snapshot)

    def merge(self, slots: dict, evidence: dict) -> list:
        """合并本轮槽位：有变化才入撤销栈；预算下界高于上界只报冲突、不改状态。
        例如“预算300以内”之后补“至少500元”，直接返回预算冲突，会话保持上一轮有效约束。

        参数：
            slots (dict): 本轮解析出的槽位。
            evidence (dict): 槽位对应的来源与置信度证据。

        返回：
            list: 冲突列表，空列表表示合并成功。
        """
        proposed = {**self.constraints, **slots}
        if proposed.get('budget_min', 0) > proposed.get('budget_max', float('inf')):
            return [{'slot': 'budget', 'message': '预算下限高于上限，请修改预算或撤销上一步'}]
        if proposed != self.constraints:
            self.history = (self.history + [(deepcopy(self.constraints), deepcopy(self.slot_evidence))])[-50:]
            self.constraints = proposed
            self.slot_evidence.update(deepcopy(evidence))
        return []

    def undo(self):
        """撤销上一轮有效修改，把约束与证据还原到入栈前；没有历史则原样返回。
        例如“撤销”对最后一轮“蓝色”生效，撤销后颜色槽位消失、预算保留。
        """
        if self.history:
            self.constraints, self.slot_evidence = self.history.pop()


class SessionStore:
    """会话表：读取时顺带清理过期会话，超过上限时淘汰最久未用的会话。
    例如 TTL 到期在下次 get 时被删除；表满时淘汰 updated_at 最小的那条。
    """

    def __init__(self, settings, clock=time.monotonic):
        """保存配置与时钟；时钟单独注入，测试里可换成固定值模拟过期。
        例如单测传入 fake_clock 就能让 session_ttl_seconds 瞬间到期。

        参数：
            settings: 提供 session_ttl_seconds、max_sessions、top_k 的配置对象。
            clock (callable): 单调时钟函数，默认 time.monotonic。
        """
        self.settings = settings
        self.clock = clock
        self._sessions = {}

    def get(self, user_id, session_id) -> SessionContext:
        """按用户+会话取上下文：先清理过期条目，不存在就新建，最后刷新活跃时间。
        例如同一用户第二条消息拿到同一份约束，不会从空会话重新开始。

        参数：
            user_id (str): 用户标识。
            session_id (str): 会话标识。

        返回：
            SessionContext: 已刷新 updated_at 的会话上下文。
        """
        now = self.clock()
        for key in list(self._sessions):
            if now - self._sessions[key].updated_at >= self.settings.session_ttl_seconds:
                del self._sessions[key]
        key = (user_id, session_id)
        if key not in self._sessions:
            if len(self._sessions) >= self.settings.max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
                del self._sessions[oldest]
            self._sessions[key] = SessionContext(user_id, session_id, top_k=self.settings.top_k)
        self._sessions[key].updated_at = now
        return self._sessions[key]

    def end(self, user_id, session_id):
        """结束会话并丢弃其上下文，之后同 key 再来会得到全新的空会话。
        例如“结束会话”后状态返回 ended，下一条消息从空约束、空偏好开始。
        """
        self._sessions.pop((user_id, session_id), None)
