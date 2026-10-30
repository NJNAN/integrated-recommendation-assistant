"""三个核心数据结构与 JSON 校验：所有跨模块数据都必须先通过这里。"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from math import isfinite

SOURCES = {'text', 'voice', 'vision', 'gesture'}
INTENTS = {'discover', 'similar', 'filter', 'compare', 'feedback', 'explain'}
SLOTS = {'category', 'budget_min', 'budget_max', 'recipient', 'color', 'scene', 'currency'}
GESTURES = {'thumbs_up', 'thumbs_down', 'open_palm', 'victory'}
PAYLOAD_KEYS = {'text', 'transcript', 'labels', 'gesture', 'item_id', 'item_ids', 'intent', 'slots',
                'action', 'feedback', 'top_k'}
CURRENCIES = ('CNY', 'GBP')


class ContractError(ValueError):
    """带错误码的输入校验异常：错误码让调用方区分失败原因。"""

    def __init__(self, message, code='INVALID_EVENT'):
        super().__init__(message)
        self.code = code


class _JSONMixin:
    """统一 to_dict：把 dataclass 转成普通 dict（可直接 JSON 序列化）。"""

    def to_dict(self):
        return asdict(self)


def _check(condition, message, code='INVALID_EVENT'):
    """条件不成立就抛带错误码的 ContractError，替代重复的 if + raise。"""
    if not condition:
        raise ContractError(message, code)


def _from_dict(cls, data, code, message, transform=None):
    """dict -> dataclass；非对象、字段缺失或非法统一映射成该结构的错误码。"""
    if not isinstance(data, dict):
        raise ContractError(message, code)
    try:
        return cls(**(transform(dict(data)) if transform else data))
    except (TypeError, ContractError) as exc:
        raise ContractError(str(exc), code) from exc


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ContractError(f'{name} 必须是非空字符串')
    return value


def number(value, name, low=0, high=float('inf')):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f'{name} 必须是 [{low}, {high}] 内的数值')
    if not isfinite(value := float(value)) or not low <= value <= high:
        raise ContractError(f'{name} 必须是 [{low}, {high}] 内的有限数值')
    return value


def validate_slots(slots):
    """槽位只允许白名单键；预算、货币、文本分别校验。"""
    _check(isinstance(slots, dict) and not set(slots) - SLOTS, 'slots 必须是对象，且只能包含支持的槽位')
    for key, value in slots.items():
        if key.startswith('budget_'):
            number(value, key)
        elif key != 'currency':
            nonempty(value, key)
        elif value not in CURRENCIES:
            raise ContractError('currency 仅支持 CNY 或 GBP')
    return slots


def _metadata(data):
    """trace_id/user_id/session_id/timestamp/source 是跨模块追踪的必备字段。"""
    for key in ('trace_id', 'user_id', 'session_id', 'timestamp', 'source'):
        nonempty(data.get(key), key)
    _check(data['source'] in SOURCES, 'source 必须是 text、voice、vision 或 gesture')
    try:
        stamp = datetime.fromisoformat(str(data['timestamp']).replace('Z', '+00:00'))
    except ValueError as exc:
        raise ContractError('timestamp 必须是包含时区的 ISO 8601 时间') from exc
    _check(stamp.tzinfo is not None, 'timestamp 必须包含时区')


def _validate_labels(labels):
    if not isinstance(labels, list) or len(labels) > 100:
        raise ContractError('labels 必须是最多 100 项的数组')
    for tag in labels:
        _check(isinstance(tag, dict), '视觉标签必须是对象')
        nonempty(tag.get('label'), 'label')
        number(tag.get('confidence'), 'confidence', 0, 1)
    return labels


@dataclass
class InteractionEvent(_JSONMixin):
    """一次用户交互：统一承载文本、语音转写、视觉标签和手势。"""
    trace_id: str; user_id: str; session_id: str
    timestamp: str; source: str; payload: dict

    @classmethod
    def from_dict(cls, data):
        _check(isinstance(data, dict), '事件必须是 JSON 对象')
        _metadata(data)
        payload = data.get('payload')
        _check(isinstance(payload, dict), 'payload 必须是对象')
        _check(not set(payload) - PAYLOAD_KEYS, 'payload 包含未知字段')
        for key in ('text', 'transcript', 'item_id'):
            if key in payload:
                nonempty(payload[key], key)
        _check('intent' not in payload or payload['intent'] in INTENTS, '不支持的 intent')
        _check('action' not in payload or payload['action'] in ('undo', 'end', 'reset'), '不支持的 action')
        _check('feedback' not in payload or
               (type(payload['feedback']) is int and payload['feedback'] in (-1, 1)), 'feedback 必须为 1 或 -1')
        if 'slots' in payload:
            validate_slots(payload['slots'])
        if 'item_ids' in payload:
            ids = payload['item_ids']
            _check(isinstance(ids, list) and len(ids) <= 24, 'item_ids 必须是最多 24 项的数组')
            for item in ids:
                nonempty(item, 'item_ids')
        _check('top_k' not in payload or
               (type(payload['top_k']) is int and 1 <= payload['top_k'] <= 24), 'top_k 必须是 1 到 24 的整数')
        if data['source'] == 'voice':
            nonempty(payload.get('transcript'), 'transcript')
        _check(data['source'] != 'gesture' or payload.get('gesture') in GESTURES, '不支持的 gesture')
        if 'labels' in payload or data['source'] == 'vision':
            _validate_labels(payload.get('labels'))
        _check(data['source'] != 'text' or any(k in payload for k in ('text', 'slots', 'intent', 'action')),
               '文本事件缺少内容')
        _check(set(data) == {'trace_id', 'user_id', 'session_id', 'timestamp', 'source', 'payload'},
               '事件包含未知字段')
        return cls(**data)


@dataclass
class RecommendationRequest(_JSONMixin):
    """跨项目推荐请求：模块一/模块三之间只通过这个结构交换数据。"""
    trace_id: str; user_id: str; session_id: str; timestamp: str
    source: str; intent: str
    constraints: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    item_ids: list = field(default_factory=list)
    top_k: int = 6

    def __post_init__(self):
        _metadata(asdict(self))
        validate_slots(self.constraints)
        _check(self.intent in INTENTS, '不支持的推荐意图', 'INVALID_REQUEST')
        _check(type(self.top_k) is int and 1 <= self.top_k <= 24,
               'top_k 必须是 1 到 24 的整数', 'INVALID_REQUEST')
        _check(isinstance(self.preferences, dict) and all(
            type(v) is int and v in (-1, 1) for v in self.preferences.values()),
            'preferences 必须为商品ID到正负反馈的映射', 'INVALID_REQUEST')
        _check(isinstance(self.item_ids, list), 'item_ids 必须为数组', 'INVALID_REQUEST')
        for item in self.item_ids:
            nonempty(item, 'item_id')

    @classmethod
    def from_dict(cls, data):
        return _from_dict(cls, data, 'INVALID_REQUEST', '推荐请求必须是对象')


@dataclass
class EvidencePack(_JSONMixin):
    """证据包：解释只能引用这里的事实，保证“每条解释都有可核验证据”。"""
    trace_id: str; item_id: str; attributes: dict
    sources: list; matched_constraints: dict
    visual_labels: list; snippets: list

    def __post_init__(self):
        nonempty(self.trace_id, 'evidence.trace_id')
        nonempty(self.item_id, 'evidence.item_id')
        _check(isinstance(self.attributes, dict) and isinstance(self.matched_constraints, dict),
               '证据属性与匹配约束必须是对象', 'INVALID_ITEM')
        _check(isinstance(self.sources, list) and all(isinstance(s, str) for s in self.sources),
               '证据来源必须是字符串数组', 'INVALID_ITEM')
        _check(isinstance(self.visual_labels, list) and isinstance(self.snippets, list),
               '视觉标签与证据片段必须是数组', 'INVALID_ITEM')
        for tag in self.visual_labels:
            nonempty(tag.get('label'), 'label')
            number(tag.get('confidence'), 'confidence', 0, 1)
        for snippet in self.snippets:
            _check(isinstance(snippet, dict), '证据片段必须是对象', 'INVALID_ITEM')
            for key in ('id', 'source', 'text'):
                nonempty(snippet.get(key), key)

    @classmethod
    def from_dict(cls, data):
        return _from_dict(cls, data, 'INVALID_ITEM', '证据包必须是对象')


@dataclass
class RecommendationItem(_JSONMixin):
    """一条推荐结果：商品事实 + 来源 + 分数明细 + 证据包 + 解释。"""
    trace_id: str; item_id: str; name: str; price: float | None
    category: str; color: str; sources: list; score: float
    score_breakdown: dict; evidence: EvidencePack
    explanation: dict = field(default_factory=dict)
    currency: str = 'CNY'; price_kind: str = 'catalog'

    def __post_init__(self):
        for key in ('trace_id', 'item_id', 'name', 'category', 'color'):
            nonempty(getattr(self, key), key)
        if self.price is None:
            _check(self.price_kind == 'missing', '缺失价格必须明确标注', 'INVALID_ITEM')
        else:
            number(self.price, 'price')
        _check(self.currency in CURRENCIES, '不支持的价格货币', 'INVALID_ITEM')
        number(self.score, 'score', -1000, 1000)
        _check(isinstance(self.sources, list) and self.sources and isinstance(self.evidence, EvidencePack),
               '商品必须保留来源和证据包', 'INVALID_CATALOG')
        _check(self.evidence.trace_id == self.trace_id and self.evidence.item_id == self.item_id,
               '商品与证据包标识不一致', 'INVALID_ITEM')
        _check(isinstance(self.score_breakdown, dict) and isinstance(self.explanation, dict),
               '分数明细与解释必须是对象', 'INVALID_ITEM')
        for key, value in self.score_breakdown.items():
            number(value, key, -1000, 1000)

    @classmethod
    def from_dict(cls, data):
        return _from_dict(cls, data, 'INVALID_ITEM', '推荐商品必须是对象',
                          lambda row: {**row, 'evidence': EvidencePack.from_dict(row.get('evidence'))})
