"""规则式多模态理解：把一条事件归一成意图、槽位、反馈和缺失项，可复现、可解释。"""
from dataclasses import dataclass, field, asdict
import re
from assistant.contracts import ContractError, InteractionEvent

CATEGORY_ALIASES = {'背包': ('背包', '双肩包', '书包'), '杯子': ('杯子', '保温杯', '马克杯', '水杯'),
                    '耳机': ('耳机',), '台灯': ('台灯',), '键盘': ('键盘',), '雨伞': ('雨伞', '晴雨伞')}
for _name in ('家居装饰', '装饰品', '生活用品', '收纳用品', '派对用品', '手工用品', '厨具', '文具用品'):
    CATEGORY_ALIASES[_name] = (_name,)
COLORS = ('蓝色', '红色', '白色', '黑色', '绿色', '黄色', '粉色')
BUDGET_HINTS = ('预算', '至少', '不低于', '最低', '不超过', '最多', '超过', '低于', '高于',
                '少于', '多于', '以内', '以下', '以上', '块钱')
_THOUSANDS = re.compile(r'(?<=\d)[,，](?=\d{3}(?:\D|$))')


@dataclass
class IntentResult:
    """承载一次事件的理解结果：意图、槽位、反馈方向、动作、冲突与缺失项。
    例如“预算300以内的蓝色背包”会得到品类、颜色和预算上限三个槽位。
    """
    intent: str = 'discover'
    slots: dict = field(default_factory=dict)
    slot_evidence: dict = field(default_factory=dict)
    item_ids: list = field(default_factory=list)
    feedback: int | None = None
    action: str | None = None
    conflicts: list = field(default_factory=list)
    ignored_labels: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    text: str = ''

    def to_dict(self):
        """把结果转成可 JSON 序列化的普通字典，方便写进响应和 trace。
        例如落盘或回包前调用 to_dict()，字段会原样变成 dict 键。
        """
        return asdict(self)


def understand(event: InteractionEvent, visual_threshold: float = .7) -> IntentResult:
    """把一条事件归一成意图、槽位、反馈与缺失项，顺序固定、结果可复现。
    例如“预算300以内的蓝色背包”会先取文字槽位，再合并视觉标签，最后判为 filter。
    抽取顺序：文本/语音 -> 视觉标签 -> 显式 slots，再按优先级判定意图与缺失项。

    参数：
        event (InteractionEvent): 待理解的一条事件，读取 source 和 payload。
        visual_threshold (float): 视觉标签置信度门槛，低于它的只记入 ignored_labels，不进槽位。

    返回：
        IntentResult: 归一后的理解结果，缺失项写在 missing 里。
    """
    payload = event.payload
    text = payload.get('transcript', payload.get('text', ''))
    result = IntentResult(text=text, action=payload.get('action'))
    text_source = 'voice' if event.source == 'voice' else 'text'

    def add(key, value, source, confidence=1.0):
        """写入一个槽位：同轮槽位冲突时保留先出现的值，并记入冲突列表。
        例如文字说“蓝色”、视觉标签说“红色”，只保留蓝色，冲突写进 conflicts。

        参数：
            key (str): 槽位名，如 category、color、budget_max。
            value: 槽位值。
            source (str): 证据来源，text/voice/vision/explicit。
            confidence (float): 该来源的置信度，显式槽位默认 1.0。
        """
        if key in result.slots and result.slots[key] != value:
            result.conflicts.append({'slot': key, 'values': [result.slots[key], value], 'source': source})
            return
        result.slots[key] = value
        result.slot_evidence[key] = {'value': value, 'source': source, 'confidence': confidence}

    if text in ('撤销', '撤销上一步'):
        result.action = 'undo'
    elif text in ('结束会话', '清空会话', '重新开始'):
        result.action = 'end' if text == '结束会话' else 'reset'
    for canonical, aliases in CATEGORY_ALIASES.items():
        if any(alias in text for alias in aliases):
            add('category', canonical, text_source)
    for color in COLORS:
        if color in text:
            add('color', color, text_source)
    for key, choices in (('recipient', ('朋友', '父母', '学生', '同事', '自己', '孩子')),
                         ('scene', ('通勤', '旅行', '学习', '运动', '办公', '居家'))):
        values = [value for value in choices if value in text]
        if values:
            add(key, values[0], text_source)
    # 预算：先去掉合法千分位；分组不合法（1,00）时不猜数字，交给缺失项追问。
    cleaned = _THOUSANDS.sub('', text)
    if not re.search(r'\d[,，]\d', cleaned):
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:到|至|[-~～])\s*(\d+(?:\.\d+)?)\s*元?', cleaned)
        if match:
            add('budget_min', float(match[1]), text_source)
            add('budget_max', float(match[2]), text_source)
        else:
            for pattern in (r'(?:至少|不低于|最低)\s*(\d+(?:\.\d+)?)',
                            r'(\d+(?:\.\d+)?)\s*元?\s*(?:以上|起)',
                            r'(?<!不要)(?<!别)(?:超过|高于|多于)\s*(\d+(?:\.\d+)?)'):
                if match := re.search(pattern, cleaned):
                    add('budget_min', float(match[1]), text_source)
            for pattern in (r'(?:预算|不超过|最多|低于|少于|上限)\s*(\d+(?:\.\d+)?)',
                            r'(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下)',
                            r'(?:不要|别)\s*(?:超过|高于)\s*(\d+(?:\.\d+)?)'):
                if match := re.search(pattern, cleaned):
                    add('budget_max', float(match[1]), text_source)
    if any(key.startswith('budget_') for key in result.slots) and any(u in text for u in ('英镑', 'GBP', '£')):
        add('currency', 'GBP', text_source)
    for label in payload.get('labels', []):
        if label['confidence'] < visual_threshold:
            result.ignored_labels.append(label)          # 低置信标签只记录，不进入槽位
            continue
        for canonical, aliases in CATEGORY_ALIASES.items():
            if label['label'] in aliases:
                add('category', canonical, 'vision', label['confidence'])
        if label['label'] in COLORS:
            add('color', label['label'], 'vision', label['confidence'])
    for key, value in payload.get('slots', {}).items():
        add(key, value, 'explicit')
    if re.search(r'(?:预算|至少|最多|不超过|不低于)\s*-\d', text):
        raise ContractError('预算不能是负数')
    if any(hint in text for hint in BUDGET_HINTS) and not any(k.startswith('budget_') for k in result.slots):
        result.missing.append('预算金额（请使用阿拉伯数字，例如300元以内）')
    if payload.get('item_id'):
        result.item_ids = [payload['item_id']]
    if payload.get('item_ids'):
        result.item_ids = list(dict.fromkeys(payload['item_ids']))
    gesture = payload.get('gesture')
    # 意图优先级：反馈 > 解释 > 比较 > 相似 > 筛选 > 发现。
    if '不喜欢' in text or gesture == 'thumbs_down':
        result.intent, result.feedback = 'feedback', -1
    elif '喜欢' in text or '点赞' in text or gesture == 'thumbs_up':
        result.intent, result.feedback = 'feedback', 1
    elif any(word in text for word in ('为什么', '解释')):
        result.intent = 'explain'
    elif any(word in text for word in ('对比', '比较')):
        result.intent = 'compare'
    elif any(word in text for word in ('相似', '类似')):
        result.intent = 'similar'
    elif result.slots or '筛选' in text or '只看' in text:
        result.intent = 'filter'
    result.intent = payload.get('intent', result.intent)
    result.feedback = payload.get('feedback', result.feedback)
    if gesture == 'open_palm':
        result.action = 'end'
    if result.intent in ('similar', 'explain', 'feedback') and not result.item_ids:
        result.missing.append('item_id')
    if result.intent == 'compare' and len(result.item_ids) < 2:
        result.missing.append('至少两个 item_ids')
    if result.intent == 'feedback' and result.feedback is None:
        result.missing.append('反馈 正负方向')
    if result.intent == 'filter' and not result.slots:
        result.missing.append('至少一个筛选条件')
    if event.source == 'vision' and not result.slots and not text and not payload.get('intent'):
        result.missing.append('可信视觉标签或补充文字')
    return result
