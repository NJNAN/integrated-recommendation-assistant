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
    """承载带错误码的输入校验异常，调用方按码决定提示还是降级。

    例如缺 trace_id 抛 INVALID_EVENT，价格越界抛 INVALID_ITEM。

    参数：
        message (str): 给人看的失败原因。
        code (str): 机器可读错误码，默认 INVALID_EVENT。
    """

    def __init__(self, message, code='INVALID_EVENT'):
        """保存失败原因和错误码，供调用方按码分支处理。

        例如 code='INVALID_EVENT' 时接口层回 400 并提示补字段。

        参数：
            message (str): 给人看的失败原因。
            code (str): 机器可读错误码。
        """
        super().__init__(message)
        self.code = code


class _JSONMixin:
    """给 dataclass 统一补 to_dict，把实例转成可 JSON 序列化的普通 dict。

    例如 InteractionEvent(...).to_dict() 能直接塞进响应体。
    """

    def to_dict(self):
        """把 dataclass 实例转成普通 dict，可以直接 JSON 序列化。

        例如 RecommendationItem(...).to_dict() 能原样写进响应体。
        """
        return asdict(self)


def _check(condition, message, code='INVALID_EVENT'):
    """条件不成立就抛 ContractError，替掉重复的 if + raise。

    例如 _check(len(ids) <= 24, 'item_ids 超长') 一行完成校验。

    参数：
        condition (bool): 必须为真的条件。
        message (str): 失败原因，直接写进异常文本。
        code (str): 错误码，默认 INVALID_EVENT。
    """
    if not condition:
        raise ContractError(message, code)


def _from_dict(cls, data, code, message, transform=None):
    """把外部 dict 还原成 dataclass，失败统一翻成该结构的错误码。

    例如推荐请求传了数组，抛 INVALID_REQUEST 而不是裸 TypeError。

    参数：
        cls (type): 目标 dataclass。
        data (dict): 待还原的原始对象。
        code (str): 失败时使用的错误码。
        message (str): 非对象时的提示。
        transform (callable | None): 可选预处理，如先还原 evidence 子结构。

    返回：
        cls: 校验通过的实例。
    """
    if not isinstance(data, dict):
        raise ContractError(message, code)
    try:
        return cls(**(transform(dict(data)) if transform else data))
    except (TypeError, ContractError) as exc:
        raise ContractError(str(exc), code) from exc


def nonempty(value, name):
    """校验并返回非空短文本，空串、纯空白或超过 4000 字直接报错。

    例如 trace_id、item_id、label 这类标识字段都走这里。

    参数：
        value (str): 待校验文本。
        name (str): 字段名，只用于拼报错信息。

    返回：
        str: 原样返回，方便调用方链式赋值。

    异常：
        ContractError: 不是字符串或为空时抛出。
    """
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ContractError(f'{name} 必须是非空字符串')
    return value


def number(value, name, low=0, high=float('inf')):
    """校验区间内的有限数值，把布尔、NaN 和 inf 全部挡掉。

    例如 confidence 传 True 或 1.5 都会被拒，只留 0 到 1 的有限实数。

    参数：
        value (int | float): 待校验数值。
        name (str): 字段名，用于报错文本。
        low (float): 下界，默认 0。
        high (float): 上界，默认无穷大。

    返回：
        float: 统一转成浮点后的数值。

    异常：
        ContractError: 越界或不是有限数时抛出。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f'{name} 必须是 [{low}, {high}] 内的数值')
    if not isfinite(value := float(value)) or not low <= value <= high:
        raise ContractError(f'{name} 必须是 [{low}, {high}] 内的有限数值')
    return value


def validate_slots(slots):
    """校验槽位对象：只认白名单键，预算、货币和文本各走各的规则。

    例如 {'budget_max': 300, 'currency': 'GBP'} 合法，{'unknown': 1} 直接报错。

    参数：
        slots (dict): 解析出的槽位。

    返回：
        dict: 原样返回，方便内联调用。
    """
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
    """校验追踪必备字段：五个标识齐全、来源合法、时间戳必须带时区。

    例如 source 写成 'camera' 会被拒，只认 text/voice/vision/gesture。

    参数：
        data (dict): 待校验的事件或请求字典。

    异常：
        ContractError: 字段缺失、来源非法或时间戳无时区时抛出。
    """
    for key in ('trace_id', 'user_id', 'session_id', 'timestamp', 'source'):
        nonempty(data.get(key), key)
    _check(data['source'] in SOURCES, 'source 必须是 text、voice、vision 或 gesture')
    try:
        stamp = datetime.fromisoformat(str(data['timestamp']).replace('Z', '+00:00'))
    except ValueError as exc:
        raise ContractError('timestamp 必须是包含时区的 ISO 8601 时间') from exc
    _check(stamp.tzinfo is not None, 'timestamp 必须包含时区')


def _validate_labels(labels):
    """校验视觉标签数组：最多 100 项，每项要有标签名和 0 到 1 的置信度。

    例如 [{'label': '红色', 'confidence': 0.92}] 合法，置信度 1.1 被拒。

    参数：
        labels (list): 视觉模型给出的标签列表。

    返回：
        list: 原样返回，方便调用方继续使用。

    异常：
        ContractError: 结构或置信度不合法时抛出。
    """
    if not isinstance(labels, list) or len(labels) > 100:
        raise ContractError('labels 必须是最多 100 项的数组')
    for tag in labels:
        _check(isinstance(tag, dict), '视觉标签必须是对象')
        nonempty(tag.get('label'), 'label')
        number(tag.get('confidence'), 'confidence', 0, 1)
    return labels


@dataclass
class InteractionEvent(_JSONMixin):
    """承载一次用户交互，把文本、语音转写、视觉标签和手势收进同一结构。

    例如语音输入必须带 transcript，视觉输入必须带 labels，否则校验失败。

    参数：
        trace_id (str): 全链路追踪 ID。
        user_id (str): 用户标识。
        session_id (str): 会话标识。
        timestamp (str): 带时区的 ISO 8601 时间。
        source (str): 输入来源，text/voice/vision/gesture 之一。
        payload (dict): 交互内容，字段受 PAYLOAD_KEYS 白名单限制。
    """
    trace_id: str; user_id: str; session_id: str
    timestamp: str; source: str; payload: dict

    @classmethod
    def from_dict(cls, data):
        """把外部 JSON 校验并还原成事件对象，失败一律抛 INVALID_EVENT。

        例如缺 trace_id、vision 事件没有 labels 都会在这里被拦下。

        参数：
            data (dict): 前端或上游传来的原始事件。

        返回：
            InteractionEvent: 校验通过的事件。
        """
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
    """承载跨项目推荐请求，模块一和模块三之间只通过它交换数据。

    例如客户端给出 discover 意图和预算槽位，就能生成一条规范请求。

    参数：
        trace_id (str): 全链路追踪 ID。
        user_id (str): 用户标识。
        session_id (str): 会话标识。
        timestamp (str): 带时区的 ISO 8601 时间。
        source (str): 请求来源，沿用事件来源枚举。
        intent (str): 推荐意图，必须是 INTENTS 之一。
        constraints (dict): 硬约束槽位，默认空。
        preferences (dict): 商品 ID 到正负反馈的映射，默认空。
        item_ids (list): 指定商品列表，默认空。
        top_k (int): 返回条数，默认 6。
    """
    trace_id: str; user_id: str; session_id: str; timestamp: str
    source: str; intent: str
    constraints: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    item_ids: list = field(default_factory=list)
    top_k: int = 6

    def __post_init__(self):
        """构造后立即自检：字段齐全、约束合法、意图与 top_k 在允许范围。

        例如 top_k=0 或 preferences 里写 0 都会在这里报 INVALID_REQUEST。
        """
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
        """从 dict 还原推荐请求，结构错误统一抛 INVALID_REQUEST。

        例如上游传数组会被拒，而不是抛出原始 TypeError。

        参数：
            data (dict): 待还原的请求对象。

        返回：
            RecommendationRequest: 校验通过的请求。
        """
        return _from_dict(cls, data, 'INVALID_REQUEST', '推荐请求必须是对象')


@dataclass
class EvidencePack(_JSONMixin):
    """承载一件商品的证据包，解释只能引用这里的事实，做到句句可核验。

    例如“红色、300 元以内”这类解释，必须能在 attributes 里找到对应字段。

    参数：
        trace_id (str): 全链路追踪 ID。
        item_id (str): 商品 ID。
        attributes (dict): 商品属性事实。
        sources (list): 证据来源列表。
        matched_constraints (dict): 命中的约束及取值。
        visual_labels (list): 视觉标签及置信度。
        snippets (list): 证据片段，每项含 id/source/text。
    """
    trace_id: str; item_id: str; attributes: dict
    sources: list; matched_constraints: dict
    visual_labels: list; snippets: list

    def __post_init__(self):
        """构造后立即自检：标识非空、属性为对象、证据片段字段齐全。

        例如 snippet 少写 text 字段会抛 INVALID_ITEM。
        """
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
        """从 dict 还原证据包，结构错误统一抛 INVALID_ITEM。

        例如 attributes 传成数组会被拦下。

        参数：
            data (dict): 待还原的证据字典。

        返回：
            EvidencePack: 校验通过的证据包。
        """
        return _from_dict(cls, data, 'INVALID_ITEM', '证据包必须是对象')


@dataclass
class RecommendationItem(_JSONMixin):
    """承载一条推荐结果：商品事实、来源、分数明细、证据包和解释。

    例如缺价格时必须把 price_kind 标成 missing，不能静默当成 0 元。

    参数：
        trace_id (str): 全链路追踪 ID。
        item_id (str): 商品 ID。
        name (str): 商品名称。
        price (float | None): 价格；None 表示缺失。
        category (str): 商品品类。
        color (str): 商品颜色。
        sources (list): 商品来源列表。
        score (float): 综合得分。
        score_breakdown (dict): 各分项得分明细。
        evidence (EvidencePack): 证据包。
        explanation (dict): 生成的解释，默认空。
        currency (str): 货币，默认 CNY。
        price_kind (str): 价格类型，默认 catalog。
    """
    trace_id: str; item_id: str; name: str; price: float | None
    category: str; color: str; sources: list; score: float
    score_breakdown: dict; evidence: EvidencePack
    explanation: dict = field(default_factory=dict)
    currency: str = 'CNY'; price_kind: str = 'catalog'

    def __post_init__(self):
        """构造后立即自检：字段非空、缺价要标注、证据包标识必须一致。

        例如 evidence.trace_id 和商品对不上，会抛 INVALID_ITEM。
        """
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
        """从 dict 还原推荐商品，先把 evidence 子结构翻成证据包再校验。

        例如上游只给 evidence 的普通 dict 也能正确还原。

        参数：
            data (dict): 待还原的商品字典。

        返回：
            RecommendationItem: 校验通过的商品。
        """
        return _from_dict(cls, data, 'INVALID_ITEM', '推荐商品必须是对象',
                          lambda row: {**row, 'evidence': EvidencePack.from_dict(row.get('evidence'))})
