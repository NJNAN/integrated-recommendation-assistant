"""一次事件 → 意图/上下文 → 召回排序 → 证据解释：编排层是唯一入口。"""
from threading import RLock
import time
import uuid
from assistant.contracts import ContractError, InteractionEvent, RecommendationRequest
from assistant.explanation import explain
from assistant.feedback import apply_feedback
from assistant.recommendation import RecommendationAdapter
from assistant.session import SessionStore
from assistant.understanding import understand


class Orchestrator:
    """加锁处理事件；每次响应都带 status、context、warnings 与可复核的 trace。"""

    def __init__(self, settings, recommender=None, store=None):
        self.settings = settings
        self.recommender = recommender or RecommendationAdapter(settings)
        self.store = store or SessionStore(settings)
        self.lock = RLock()

    def handle(self, data):
        """线程安全地处理一条事件字典，返回统一响应。"""
        with self.lock:
            return self._handle(data)

    def _handle(self, data):
        start = time.perf_counter()
        trace_id = data.get('trace_id') if isinstance(data, dict) and isinstance(data.get('trace_id'), str) else str(uuid.uuid4())
        response = {'trace_id': trace_id, 'status': 'error', 'message': '', 'items': [], 'context': {},
                    'degraded': False, 'warnings': [], 'error': None, 'trace': []}

        def record(stage, value):
            response['trace'].append({'trace_id': response['trace_id'], 'stage': stage,
                                      'elapsed_ms': round((time.perf_counter() - start) * 1000, 3), 'data': value})

        def finish(status, message):
            response['status'], response['message'] = status, message
            response['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 3)
            record('result', {'status': status, 'item_ids': [item['item_id'] for item in response['items']]})
            return response

        ctx = snapshot = None
        try:
            event = InteractionEvent.from_dict(data)
            response['trace_id'] = event.trace_id
            record('event', event.to_dict())
            intent = understand(event, self.settings.visual_threshold)
            record('intent', intent.to_dict())
            ctx = self.store.get(event.user_id, event.session_id)
            snapshot = ctx.snapshot()
            response['context'] = ctx.public()
            if intent.action in ('end', 'reset'):
                self.store.end(event.user_id, event.session_id)
                response['context'] = {}
                record('context', {})
                return finish('ended', '会话已清理，可以重新开始。')
            if intent.action == 'undo':
                ctx.undo()
            if intent.conflicts:
                response['conflicts'] = intent.conflicts
                record('context', ctx.public())
                return finish('conflict', '同一轮输入包含矛盾条件，请确认品类、颜色或预算。')
            catalog = self.recommender.catalog
            if any(key.startswith('budget_') for key in intent.slots):
                currency = intent.slots.get('currency', ctx.constraints.get('currency', 'CNY'))
                if ('元' in intent.text and '英镑' not in intent.text) or '人民币' in intent.text:
                    currency = 'CNY'
                if currency != catalog.currency:
                    response['missing_slots'] = ['与商品目录一致的预算货币']
                    record('context', ctx.public())
                    message = ('目录价格为 GBP 历史单价。请明确使用英镑预算，例如“预算10英镑以内”；不会将人民币直接当作英镑。'
                               if catalog.currency == 'GBP' else '当前目录使用人民币，请使用人民币预算。')
                    return finish('needs_clarification', message)
            if intent.missing:
                missing = [name for name in intent.missing if not (name == '至少一个筛选条件' and ctx.constraints)]
                if missing:
                    response['missing_slots'] = missing
                    record('context', ctx.public())
                    return finish('needs_clarification', '请补充：' + '、'.join(missing))
            conflicts = ctx.merge(intent.slots, intent.slot_evidence)
            if conflicts:
                response['conflicts'] = conflicts
                record('context', ctx.public())
                return finish('conflict', conflicts[0]['message'])
            if intent.intent == 'feedback':
                if intent.slots:
                    raise ContractError('反馈事件不能同时修改筛选条件', 'INVALID_FEEDBACK')
                if not intent.item_ids or intent.feedback not in (-1, 1):
                    raise ContractError('反馈必须指定商品和正负方向', 'INVALID_FEEDBACK')
                apply_feedback(ctx, intent.item_ids[0], intent.feedback)
                active_intent, targets = ctx.last_intent, list(ctx.target_item_ids)
            else:
                active_intent = ctx.last_intent if intent.action == 'undo' else intent.intent
                targets = list(ctx.target_item_ids if intent.action == 'undo' else intent.item_ids)
                for target in targets:
                    catalog.get(target)
                ctx.last_intent, ctx.target_item_ids = active_intent, list(targets)
            ctx.top_k = event.payload.get('top_k', ctx.top_k)
            request = RecommendationRequest(event.trace_id, event.user_id, event.session_id, event.timestamp,
                                            event.source, active_intent, dict(ctx.constraints), dict(ctx.preferences),
                                            list(targets), ctx.top_k)
            response['context'] = ctx.public()
            record('context', ctx.public())
            record('request', request.to_dict())
            items, diagnostic = self.recommender.recommend(request)
            record('recall_and_ranking', diagnostic)
            response['warnings'] = diagnostic['warnings']
            response['degraded'] = bool(diagnostic['warnings'])
            response['mode'] = diagnostic['mode']
            response['dataset'] = catalog.dataset
            response['currency'] = catalog.currency
            response['price_notice'] = catalog.price_notice
            for item in items:
                item.explanation = explain(item)
            response['items'] = [item.to_dict() for item in items]
            ctx.last_item_ids = [item.item_id for item in items]
            response['context'] = ctx.public()
            record('explanation', [{'item_id': item.item_id, 'explanation': item.explanation} for item in items])
            if not items:
                return finish('empty', '没有符合当前品类、预算和颜色的商品。请调整条件或撤销上一步；硬约束保持有效。')
            if active_intent == 'compare':
                message = '已按相同目录事实比较所选商品，价格与属性见下方。'
            elif intent.intent == 'feedback':
                message = '已记录当前会话反馈，并按原条件重新排序。'
            else:
                message = f'找到 {len(items)} 件符合条件的商品。'
            return finish('ok', message)
        except ContractError as exc:
            response['error'] = {'code': exc.code, 'message': str(exc)}
            if ctx is not None:
                if snapshot is not None:
                    ctx.restore(snapshot)
                response['context'] = ctx.public()
            record('error', response['error'])
            return finish('error', str(exc))
        except Exception:
            response['error'] = {'code': 'INTERNAL_ERROR', 'message': '系统处理失败，请重试'}
            if ctx is not None:
                if snapshot is not None:
                    ctx.restore(snapshot)
                response['context'] = ctx.public()
            record('error', response['error'])
            return finish('error', response['error']['message'])
