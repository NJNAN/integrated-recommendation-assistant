"""一次事件 → 意图/上下文 → 召回排序 → 证据解释：编排层是唯一入口。

讲解顺序：
1. `handle` 加锁串行；`_handle` 是真正的 12 步主链路（每一步都有编号注释）。
2. 三个内部小函数：`record`（记一条 trace）、`finish`（统一收尾）、快照/回滚。
3. 六种 status：ok / empty / needs_clarification / conflict / ended / error。
4. 两条异常路径：ContractError（业务/校验错 → 400）与 Exception（兜底 → 500），都会回滚会话。

阅读约定：每个函数的 docstring 都写了「参数（每个参数是什么、从哪来、取值）」「返回」「异常」。
"""
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
    """编排器：唯一入口。依赖通过构造注入，测试时可以换成假上游 / 假会话。"""

    def __init__(self, settings, recommender=None, store=None):
        """装配三个依赖，并准备一把锁。

        参数：
            settings (config.settings.Settings): 只读配置，本类用到：
                - `visual_threshold`：传给 understand()，决定多低置信度的视觉标签被忽略；
                - 其余路径/端口/超时等由下面两个依赖各自读取。
            recommender (可选，默认 None): 推荐器，必须有 `catalog` 属性和 `recommend(request)` 方法。
                - 传 None：用 `RecommendationAdapter(settings)`（正常路径）；
                - 传具体对象：用传入的。测试/阶段3 用它注入「假上游」（例如必超时的 TimeoutRemote 包进去）。
            store (可选，默认 None): 会话存储，必须有 `get(user_id, session_id)` 和 `end(...)`。
                - 传 None：用 `SessionStore(settings)`（内存会话 + TTL）；
                - 传具体对象：用传入的（测试可以给一个可控时钟的 store）。

        副作用：
            `self.lock`：可重入锁（RLock）。HTTP 是多线程的，加锁后同一个进程内一次只处理一个事件，
            避免两个用户的会话状态互相覆盖。
        """
        self.settings = settings
        self.recommender = recommender or RecommendationAdapter(settings)
        self.store = store or SessionStore(settings)
        self.lock = RLock()          # HTTP 是多线程的：同一进程内串行处理，避免会话状态互相打架

    def handle(self, data):
        """对外唯一入口：线程安全地处理一条事件，返回统一响应。

        参数：
            data (dict): 原始事件 JSON，字段见 contracts.InteractionEvent：
                trace_id / user_id / session_id / timestamp / source / payload。
                可能是非法输入（缺字段、类型不对），这里不做校验，交给 `_handle` 里的契约校验。

        返回：
            dict: 统一响应，固定字段：
            trace_id / status / message / items / context / degraded / warnings / error / trace / elapsed_ms。

        异常：
            不向外抛异常。业务错误变成 `status='error'` + `error.code`；
            未预期异常变成 `status='error'` + `error.code='INTERNAL_ERROR'`。
        """
        with self.lock:
            return self._handle(data)

    def _handle(self, data):
        """主链路（12 步）。调用方应当是 `handle`（它负责加锁）。

        参数：
            data (dict): 同 `handle` 的 data——原始事件字典。

        返回：
            dict: 统一响应。`status` 只会是六种之一：
            - 'ok'：有结果；
            - 'empty'：硬约束把所有候选都筛掉了（不放宽条件）；
            - 'needs_clarification'：缺必选槽位/币种不符，需要用户补充；
            - 'conflict'：同一轮里条件互相矛盾，或预算上下界矛盾；
            - 'ended'：用户说结束/重置，会话已清理；
            - 'error'：校验失败或内部异常（带 `error.code`）。

        说明：
            进入后先存一份会话快照（snapshot），任何一步抛 ContractError 都恢复到这份快照，
            所以「失败的请求不会污染会话」。上游问题不算失败，只写 `warnings` + `degraded=true`。
        """
        start = time.perf_counter()

        # 事件可能连 trace_id 都没有（非法输入）：先用传入值/新 UUID 兜底，
        # 保证“即使请求非法，响应也能带着 trace_id 被追踪”。
        trace_id = data.get('trace_id') if isinstance(data, dict) and isinstance(data.get('trace_id'), str) else str(uuid.uuid4())
        response = {'trace_id': trace_id, 'status': 'error', 'message': '', 'items': [], 'context': {},
                    'degraded': False, 'warnings': [], 'error': None, 'trace': []}

        def record(stage, value):
            """记一条 trace（流水账）。

            参数：
                stage (str): 阶段名，只允许这几种：
                    'event' / 'intent' / 'context' / 'request' / 'recall_and_ranking'
                    / 'explanation' / 'result' / 'error'。
                value (任意可 JSON 序列化的数据): 这个阶段当时的数据，例如
                    事件字典、意图字典、请求字典、召回诊断信息、解释列表。

            返回：
                None。副作用是往 `response['trace']` 末尾追加一条：
                {'trace_id': 与响应一致, 'stage': 阶段名, 'elapsed_ms': 相对本次请求开始, 'data': value}。
            """
            response['trace'].append({'trace_id': response['trace_id'], 'stage': stage,
                                      'elapsed_ms': round((time.perf_counter() - start) * 1000, 3), 'data': value})

        def finish(status, message):
            """统一收尾：写状态、写文案、记总耗时，并补一条 result 阶段的 trace。

            参数：
                status (str): 六种状态之一（ok/empty/needs_clarification/conflict/ended/error）。
                message (str): 给用户看的中文提示（例如「找到 2 件符合条件的商品。」）。

            返回：
                dict: 同一个 `response` 对象（已经填好 status/message/elapsed_ms/trace[result]）。
            """
            response['status'], response['message'] = status, message
            response['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 3)
            record('result', {'status': status, 'item_ids': [item['item_id'] for item in response['items']]})
            return response

        ctx = snapshot = None
        try:
            # ① 校验事件：缺字段 / 枚举错 / 模态 payload 错 → INVALID_EVENT。
            event = InteractionEvent.from_dict(data)
            response['trace_id'] = event.trace_id
            record('event', event.to_dict())

            # ② 理解意图（成员2 的规则式抽取）：这里只消费结果，不做 NLP。
            intent = understand(event, self.settings.visual_threshold)
            record('intent', intent.to_dict())

            # ③ 取会话并立刻存快照：后面任何一步失败都靠它回滚，失败请求不污染会话。
            ctx = self.store.get(event.user_id, event.session_id)
            snapshot = ctx.snapshot()
            response['context'] = ctx.public()

            # ④ 结束/重置：清会话，返回 ended。
            if intent.action in ('end', 'reset'):
                self.store.end(event.user_id, event.session_id)
                response['context'] = {}
                record('context', {})
                return finish('ended', '会话已清理，可以重新开始。')

            # ⑤ 撤销：恢复上一轮约束（历史栈在 SessionContext 里）。
            if intent.action == 'undo':
                ctx.undo()

            # ⑥ 同一轮输入自相矛盾（一句话里既说蓝色又说红色）→ 拒绝更新。
            if intent.conflicts:
                response['conflicts'] = intent.conflicts
                record('context', ctx.public())
                return finish('conflict', '同一轮输入包含矛盾条件，请确认品类、颜色或预算。')

            # ⑦ 预算币种必须和目录一致：真实目录是 GBP，人民币预算要澄清，不能 1:1 当英镑用。
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

            # ⑧ 必选槽位缺失（如“找类似的”没给商品）→ 追问不猜；已有约束可以满足后续 filter。
            if intent.missing:
                missing = [name for name in intent.missing if not (name == '至少一个筛选条件' and ctx.constraints)]
                if missing:
                    response['missing_slots'] = missing
                    record('context', ctx.public())
                    return finish('needs_clarification', '请补充：' + '、'.join(missing))

            # ⑨ 合并进会话：预算下限 > 上限视为冲突，保持原约束不变。
            conflicts = ctx.merge(intent.slots, intent.slot_evidence)
            if conflicts:
                response['conflicts'] = conflicts
                record('context', ctx.public())
                return finish('conflict', conflicts[0]['message'])

            # ⑩ 反馈分支：覆盖当前会话偏好（不累加），随后按原条件重新排序。
            if intent.intent == 'feedback':
                if intent.slots:
                    raise ContractError('反馈事件不能同时修改筛选条件', 'INVALID_FEEDBACK')
                if not intent.item_ids or intent.feedback not in (-1, 1):
                    raise ContractError('反馈必须指定商品和正负方向', 'INVALID_FEEDBACK')
                apply_feedback(ctx, intent.item_ids[0], intent.feedback)
                active_intent, targets = ctx.last_intent, list(ctx.target_item_ids)
            else:
                # 非反馈分支：撤销时沿用上一轮意图/目标，否则用本轮结果。
                active_intent = ctx.last_intent if intent.action == 'undo' else intent.intent
                targets = list(ctx.target_item_ids if intent.action == 'undo' else intent.item_ids)
                # 目标商品必须在目录里：不存在抛 ITEM_NOT_FOUND（在写会话状态之前就校验）。
                for target in targets:
                    catalog.get(target)
                ctx.last_intent, ctx.target_item_ids = active_intent, list(targets)

            # ⑪ 组装契约里的推荐请求 → 调推荐模块 → 逐条生成解释。
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
            response['degraded'] = bool(diagnostic['warnings'])       # 有警告就算降级（离线也算一种）
            response['mode'] = diagnostic['mode']
            response['dataset'] = catalog.dataset
            response['currency'] = catalog.currency
            response['price_notice'] = catalog.price_notice
            for item in items:
                item.explanation = explain(item)                       # 解释只能读 EvidencePack
            response['items'] = [item.to_dict() for item in items]
            ctx.last_item_ids = [item.item_id for item in items]        # 反馈只能针对这一批商品
            response['context'] = ctx.public()
            record('explanation', [{'item_id': item.item_id, 'explanation': item.explanation} for item in items])

            # ⑫ 收尾：空集不放宽硬约束；对比 / 反馈 / 普通各有自己的文案。
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
            # 业务/校验错误：带错误码返回（HTTP 层映射 400），并把会话回滚到进入时的快照。
            response['error'] = {'code': exc.code, 'message': str(exc)}
            if ctx is not None:
                if snapshot is not None:
                    ctx.restore(snapshot)
                response['context'] = ctx.public()
            record('error', response['error'])
            return finish('error', str(exc))
        except Exception:
            # 未预期异常：不把内部细节抛给调用方，统一 INTERNAL_ERROR（HTTP 层映射 500）。
            response['error'] = {'code': 'INTERNAL_ERROR', 'message': '系统处理失败，请重试'}
            if ctx is not None:
                if snapshot is not None:
                    ctx.restore(snapshot)
                response['context'] = ctx.public()
            record('error', response['error'])
            return finish('error', response['error']['message'])
