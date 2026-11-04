"""反馈闭环：只允许对当前会话最近展示的商品做正负反馈。"""
from assistant.contracts import ContractError


def apply_feedback(session, item_id, sign):
    """把一条点赞或点踩写入会话偏好，同一商品覆盖而不累加。
    例如连点两次“喜欢”仍记 +1；反馈没展示过的商品会报 INVALID_FEEDBACK。
    """
    if type(sign) is not int or sign not in (-1, 1):
        raise ContractError('反馈必须为 1（喜欢）或 -1（不喜欢）', 'INVALID_FEEDBACK')
    if item_id not in session.last_item_ids:
        raise ContractError('只能反馈当前会话最近展示的商品', 'INVALID_FEEDBACK')
    session.preferences[item_id] = sign
    return sign
