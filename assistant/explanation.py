"""证据约束解释：只引用 item.evidence 里的事实；证据不足就降级，绝不编造。"""
from assistant.contracts import EvidencePack, RecommendationItem

SYMBOLS = {'GBP': '£', 'CNY': '¥'}
UNITS = {'GBP': '英镑', 'CNY': '元'}
SLOT_NAMES = {'category': '品类', 'color': '颜色', 'recipient': '对象', 'scene': '场景'}


def _degraded(code, text):
    """降级返回：引用必须清空，保证异常时不会凭空生成证据。
    例如证据包为空时返回 INSUFFICIENT_EVIDENCE，citations 一定是空列表。
    """
    return {'text': text, 'speech_text': text, 'citations': [], 'degraded': True, 'code': code}


def _price_text(item, attributes):
    """价格短语只来自证据属性；价格缺失时明确说明，不猜数字。
    例如 10 英镑写成“历史观测单价 £10”，价格是 None 就写“价格缺失”。
    """
    price = attributes['price']
    if price is None:
        return '价格缺失'
    label = '历史观测单价' if item.price_kind == 'historical_last_observed' else '价格'
    return f"{label} {SYMBOLS.get(item.currency, '¥')}{price:g}"


def _condition_text(item, matched):
    """把命中的硬约束翻译成答辩时可以逐条讲的条件说明。
    例如预算 300 元以内、蓝色，会输出“符合当前条件：预算不超过300元、颜色为蓝色。”
    """
    parts = []
    for key, value in matched.items():
        if key == 'budget_max':
            parts.append(f"预算不超过{value:g}{UNITS.get(item.currency, '元')}")
        elif key == 'budget_min':
            parts.append(f"预算至少{value:g}{UNITS.get(item.currency, '元')}")
        else:
            parts.append(f'{SLOT_NAMES.get(key, key)}为{value}')
    return '符合当前条件：' + '、'.join(parts) + '。'


def explain(item: RecommendationItem) -> dict:
    """为一条推荐生成可核验解释；任何不一致都返回降级结果与错误码。
    规则：只引用 item.evidence 的事实，证据不足给 INSUFFICIENT_EVIDENCE，对不上给 EVIDENCE_MISMATCH。
    """
    if not isinstance(item, RecommendationItem):
        return _degraded('INVALID_ITEM', '无法生成解释：输入不是推荐商品对象。')
    pack = item.evidence
    if not isinstance(pack, EvidencePack):
        return _degraded('INVALID_ITEM', '无法生成解释：商品缺少有效证据包。')
    if not pack.snippets or not pack.attributes:
        return _degraded('INSUFFICIENT_EVIDENCE', '该商品缺少可核验证据，暂时无法生成推荐解释。')
    prefix = f'catalog:{item.item_id}:'
    if any(not str(snippet.get('id', '')).startswith(prefix) for snippet in pack.snippets):
        return _degraded('EVIDENCE_MISMATCH', '证据片段不属于该商品，已暂停解释，请核验证据来源。')
    attributes = pack.attributes
    if any(key not in attributes or attributes[key] != getattr(item, key)
           for key in ('name', 'price', 'category', 'color')):
        return _degraded('EVIDENCE_MISMATCH', '商品属性与证据不一致，已暂停解释，请核验目录。')
    text = f"{attributes['name']}，{_price_text(item, attributes)}，{attributes['color']}{attributes['category']}。"
    if pack.matched_constraints:
        text += _condition_text(item, pack.matched_constraints)
    description = [s for s in pack.snippets if str(s.get('id', '')).endswith(':description')]
    if description:
        text += description[0]['text']
    feedback = item.score_breakdown.get('feedback', 0)
    if feedback:
        text += '当前会话的' + ('正' if feedback > 0 else '负') + '反馈影响了排序。'
    return {'text': text, 'speech_text': text, 'citations': [s['id'] for s in pack.snippets],
            'degraded': False, 'code': 'OK'}
