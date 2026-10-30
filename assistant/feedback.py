"""反馈闭环：只允许对当前会话最近展示的商品做正负反馈。"""
from assistant.contracts import ContractError


def apply_feedback(session, item_id, sign):
    """覆盖式写入偏好：同方向重复反馈不累加权重，非法目标抛 INVALID_FEEDBACK。"""
    if type(sign) is not int or sign not in (-1, 1):
        raise ContractError('反馈必须为 1（喜欢）或 -1（不喜欢）', 'INVALID_FEEDBACK')
    if item_id not in session.last_item_ids:
        raise ContractError('只能反馈当前会话最近展示的商品', 'INVALID_FEEDBACK')
    session.preferences[item_id] = sign
    return sign
