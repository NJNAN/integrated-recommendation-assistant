"""按用户+会话隔离的短期上下文：约束可撤销，原始音视频一律不落库。"""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import time


@dataclass
class SessionContext:
    """一个会话的可恢复快照：约束、证据、偏好、最近展示商品与撤销栈。"""
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
        """对外只暴露约束、偏好等字段，不泄露撤销栈和时间戳。"""
        data = asdict(self)
        del data['history'], data['updated_at']
        return data

    def snapshot(self):
        return deepcopy(self.__dict__)

    def restore(self, snapshot):
        self.__dict__.clear()
        self.__dict__.update(snapshot)

    def merge(self, slots: dict, evidence: dict) -> list:
        """合并本轮槽位；预算下界高于上界时返回冲突且不修改状态。"""
        proposed = {**self.constraints, **slots}
        if proposed.get('budget_min', 0) > proposed.get('budget_max', float('inf')):
            return [{'slot': 'budget', 'message': '预算下限高于上限，请修改预算或撤销上一步'}]
        if proposed != self.constraints:
            self.history = (self.history + [(deepcopy(self.constraints), deepcopy(self.slot_evidence))])[-50:]
            self.constraints = proposed
            self.slot_evidence.update(deepcopy(evidence))
        return []

    def undo(self):
        """撤销上一轮有效修改；没有历史时保持当前状态。"""
        if self.history:
            self.constraints, self.slot_evidence = self.history.pop()


class SessionStore:
    """会话表：读取时顺带清理过期会话，超过上限时淘汰最久未用的会话。"""

    def __init__(self, settings, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self._sessions = {}

    def get(self, user_id, session_id) -> SessionContext:
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
        self._sessions.pop((user_id, session_id), None)
