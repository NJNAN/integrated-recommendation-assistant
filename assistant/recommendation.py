"""阶段3 核心：商品目录 → 多路召回 → 硬约束过滤 → 六项加权打分 → 稳定排序 → TopK + 证据包。

讲解顺序（答辩时按这个顺序指着代码讲）：
1. `Catalog.__init__`  读取并校验商品目录，脏数据直接报 INVALID_CATALOG，不带着问题往下跑。
2. `_recall_upstream`  调模块一真实推荐服务；失败只记 warnings 并降级，不抛异常给编排层。
3. `_hard_failures`    品类/颜色/预算/缺失价格四个硬闸门，判定发生在打分之前。
4. `_score_breakdown`  六项加权（热门/上下文/相似/上游/反馈/标签亲和），每项都留痕。
5. `_evidence`         组装 EvidencePack，解释模块只允许引用这里的片段。
6. `recommend`         把 1~5 串起来的五步主流程，返回 (TopK 商品, 诊断信息)。
"""
import json
from pathlib import Path
from assistant.adapters import AdapterError, HTTPRecommender
from assistant.contracts import CURRENCIES, ContractError, EvidencePack, RecommendationItem, number, nonempty


class Catalog:
    """商品目录：启动时一次性校验并建索引，之后只读。

    目录 JSON 外层字段：dataset（数据集名）、currency（币种）、source_kind（'module_one_real'
    表示来自模块一真实目录）、price_notice（价格口径说明）、items（商品数组）。
    """

    def __init__(self, path):
        try:
            document = json.loads(Path(path).read_text(encoding='utf-8'))
            rows = document['items']
            if not isinstance(rows, list) or not rows:
                raise ValueError('商品目录为空')

            # 目录级元信息：currency 决定“人民币预算要不要澄清”，real 决定打分与证据的差异。
            self.dataset = document.get('dataset', 'external-catalog')
            self.currency = document.get('currency', 'CNY')
            self.real = document.get('source_kind') == 'module_one_real'
            self.price_notice = document.get('price_notice', '')
            if self.currency not in CURRENCIES:
                raise ValueError('目录货币必须是 CNY 或 GBP')

            self.items = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('商品必须是对象')
                # 四个必备事实：没有它们，过滤和解释都无从谈起。
                for key in ('item_id', 'name', 'category', 'color'):
                    nonempty(row.get(key), key)

                # 价格允许为空，但只有“真实目录 + 明确标注 missing”才允许；
                # 合成目录里出现空价说明数据坏了，必须报错。
                if row.get('price') is None:
                    if not self.real or row.get('price_kind') != 'missing':
                        raise ValueError(f"商品 {row.get('item_id')} 缺失价格必须来自真实目录并标注 missing")
                else:
                    number(row['price'], 'price')

                number(row.get('popularity'), 'popularity', 0, 1)   # 归一化热度，必须落在 [0,1]
                if not isinstance(row.get('tags', []), list) or any(
                        not isinstance(tag, str) for tag in row.get('tags', [])):
                    raise ValueError('tags 必须是字符串数组')
                if not isinstance(row.get('description', ''), str):
                    raise ValueError('description 必须是字符串')
                for label in row.get('visual_labels', []):          # 视觉标签：label + 置信度
                    nonempty(label.get('label'), 'label')
                    number(label.get('confidence'), 'confidence', 0, 1)

                if row['item_id'] in self.items:
                    raise ValueError(f"商品ID重复：{row['item_id']}")

                # 统一补默认值，后续打分/证据代码就不必到处判空。
                row['tags'] = list(row.get('tags', []))
                row['description'] = row.get('description', '')
                row['visual_labels'] = list(row.get('visual_labels', []))
                self.items[row['item_id']] = row
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # 所有目录问题收敛成一个明确错误码：调用方能区分“目录坏了”和“没有符合条件的商品”。
            raise ContractError(f'商品目录读取或校验失败：{exc}', 'INVALID_CATALOG') from exc

    def get(self, item_id):
        """按 ID 取商品；不存在时抛 ITEM_NOT_FOUND，而不是返回 None 让人猜。"""
        if item_id not in self.items:
            raise ContractError(f'商品不存在：{item_id}', 'ITEM_NOT_FOUND')
        return self.items[item_id]


class RecommendationAdapter:
    """推荐适配器：召回 → 过滤 → 打分 → 组装证据。

    构造时可注入 `remote`（默认按 settings.recommender_url 建 HTTPRecommender）；
    测试和阶段3 的“上游超时降级”用例就是注入一个必抛错的假 remote。
    """

    def __init__(self, settings, remote=None):
        self.settings = settings
        self.catalog = Catalog(settings.catalog_path)
        # remote 为 None 表示纯离线：只靠目录召回，响应会带 OFFLINE_DEMO 警告。
        self.remote = remote if remote is not None else (HTTPRecommender(settings) if settings.recommender_url else None)

    def _recall_upstream(self, request):
        """调模块一真实推荐接口，只取候选 ID 和分数。

        返回 (分数表, warnings, mode)：
        - 没配上游：({}, [OFFLINE_DEMO], 'offline_demo')
        - 成功：({item_id: 分数}, [], 'hybrid')
        - 超时/不可用/格式错：({}, [错误码], 'offline_demo')  ← 只降级，不抛给编排层
        """
        if self.remote is None:
            return {}, [{'code': 'OFFLINE_DEMO', 'message': '未配置真实推荐服务，使用离线目录召回'}], 'offline_demo'
        try:
            rows = self.remote.recall(request)
            if not isinstance(rows, list):
                raise ContractError('上游候选必须是数组', 'INVALID_UPSTREAM')
            scores = {}
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get('item_id'), str):
                    raise ContractError('上游候选缺少商品ID', 'INVALID_UPSTREAM')
                score = number(row.get('score'), 'upstream score', -1000, 1000)
                # 同一商品被上游返回多次时取最高分，避免重复计入候选。
                if row['item_id'] in self.catalog.items:
                    scores[row['item_id']] = max(score, scores.get(row['item_id'], -1000))
            # 上游给了候选但一个都不在本目录：不能拿它的分数编造价格属性，只能报协议不匹配。
            if rows and not scores:
                raise ContractError('上游商品ID与配置目录不匹配，不能编造价格或属性', 'INVALID_UPSTREAM')
            return scores, [], 'hybrid'
        except (AdapterError, ContractError, OSError, TimeoutError) as exc:
            # 上游任何失败都转成 warnings：响应里 degraded=true、前端提示“已降级”，但结果照常返回。
            code = 'INVALID_UPSTREAM' if isinstance(exc, ContractError) else getattr(exc, 'code', 'UPSTREAM_UNAVAILABLE')
            return {}, [{'code': code, 'message': str(exc)}], 'offline_demo'

    @staticmethod
    def _hard_failures(product, constraints):
        """四个硬约束：品类、颜色、预算、缺失价格。返回失败原因列表，空列表表示通过。

        这一步在打分之前执行，所以“上游分数高 / 热门”都不能让不合规商品过关。
        """
        # 品类、颜色是精确匹配；别名和模糊说法在意图层已经归一化成标准值了。
        failures = [key for key in ('category', 'color') if key in constraints and product[key] != constraints[key]]
        price = product['price']
        if price is None and any(key in constraints for key in ('budget_min', 'budget_max')):
            # 价格缺失 + 用户给了预算：没有事实依据，不能进候选。
            failures.append('missing_price')
        elif price is not None and (price < constraints.get('budget_min', 0)
                                    or price > constraints.get('budget_max', float('inf'))):
            failures.append('budget')
        return failures

    def _score_breakdown(self, request, product, targets, upstream, mode, constraints):
        """六项加权打分。权重是答辩必问项，改任何数值都会直接影响阶段5 的反馈排名对照。

        - popularity   0.4×热度；真实目录 hybrid 模式置 0（避免合成热度干扰真实排序）
        - context      0.2×(对象命中 + 场景命中)，对象/场景是软偏好
        - similarity   0.3×同品类 + 0.1×共同标签数，只在 similar 意图计算
        - upstream     0.3×上游分数；真实目录 hybrid 模式用 1.0（以模块一排序为主）
        - feedback     2.5×该商品偏好(±1)，当前会话点赞/倒踩
        - tag_affinity 0.06×共同标签数×符号，喜欢一个商品会稍微带动同类
        """
        tags = product['tags']

        # 热门项：真实目录 hybrid 时不参与，避免两套热度口径混在一起。
        popularity = 0 if self.catalog.real and mode == 'hybrid' else .4 * product['popularity']

        # 上下文项：对象（朋友/父母…）和场景（通勤/学习…）命中几个就得几分 ×0.2。
        context = .2 * sum(constraints.get(key) in tags for key in ('recipient', 'scene'))

        # 相似项：同品类 0.3，每共享一个标签 0.1；只在“找相似”时才有意义。
        similarity = 0
        if request.intent == 'similar' and targets:
            similarity = .3 * (product['category'] == targets[0]['category'])
            similarity += .1 * len(set(tags) & set(targets[0]['tags']))

        # 上游项：普通模式 0.3；真实目录 hybrid 模式 1.0（此时候选只来自上游）。
        upstream_weight = 1 if self.catalog.real else .3

        # 标签亲和：对每个被反馈过的商品按“共同标签数 × 0.06 × 符号”累加；不计算它自己。
        affinity = sum(sign * .06 * len(set(self.catalog.items[liked]['tags']) & set(tags))
                       for liked, sign in request.preferences.items()
                       if liked in self.catalog.items and liked != product['item_id'])

        # round 到 6 位：浮点累加会留 0.36800000000000005 这类尾巴，round 后 JSON 才能稳定比对。
        return {'popularity': round(popularity, 6), 'context': round(context, 6),
                'similarity': round(similarity, 6), 'upstream': round(upstream_weight * upstream.get(product['item_id'], 0), 6),
                'feedback': round(2.5 * request.preferences.get(product['item_id'], 0), 6),
                'tag_affinity': round(affinity, 6)}

    def _evidence(self, request, product, sources, constraints):
        """组装证据包：解释模块只能引用这里的事实，所以事实必须完整、可核对。"""
        item_id = product['item_id']

        # 商品属性：解释模块会把它和 RecommendationItem 上的同名字段逐项比对，不一致就停止解释。
        attributes = {'name': product['name'], 'price': product['price'], 'category': product['category'],
                      'color': product['color'], 'currency': product.get('currency', self.catalog.currency),
                      'price_kind': product.get('price_kind', 'catalog')}

        # 片段 id 统一以 catalog:<商品ID>: 开头：解释模块靠它校验“引用是不是本商品的证据”。
        snippets = [{'id': f'catalog:{item_id}:attributes', 'source': 'catalog',
                     'text': json.dumps(attributes, ensure_ascii=False)}]
        if product['description']:
            snippets.append({'id': f'catalog:{item_id}:description', 'source': 'catalog', 'text': product['description']})
        if self.catalog.real:
            # 真实目录额外附来源：价格来自哪个文件哪一天、颜色来自哪个字段，便于追溯。
            provenance = {key: product.get(key) for key in ('price_source', 'color_source', 'original_category')}
            snippets.append({'id': f'catalog:{item_id}:provenance', 'source': 'module_one_catalog',
                             'text': json.dumps(provenance, ensure_ascii=False)})

        # 视觉标签只保留达到阈值的（默认 0.7）：低置信标签降权 = 直接忽略，不进排序也不进证据。
        labels = [tag for tag in product['visual_labels'] if tag['confidence'] >= self.settings.visual_threshold]

        # 命中的约束：品类/颜色/预算，以及“约束值出现在商品标签里”的对象/场景，用于解释里的“符合当前条件”。
        matched = {key: value for key, value in constraints.items()
                   if key in ('category', 'color', 'budget_min', 'budget_max') or value in product['tags']}
        return EvidencePack(request.trace_id, item_id, attributes, sorted(sources), matched, labels, snippets)

    def recommend(self, request):
        """主流程（五步）。返回 (按分数降序的 TopK 商品, diagnostic)。

        diagnostic 固定 6 个键，会进 trace 并出现在 stage03_demo.json：
        mode / warnings / recall_count / filtered_count / rejected / candidates。
        """
        catalog, constraints = self.catalog, request.constraints

        # ① 上游召回：可能成功、可能是 OFFLINE_DEMO、也可能超时降级（只记 warnings）。
        upstream, warnings, mode = self._recall_upstream(request)

        # ② 多路召回：popularity 先把全目录放进候选，再按条件叠加其它来源标记。
        routes = {item_id: {'popularity'} for item_id in catalog.items}
        for item_id in upstream:
            routes[item_id].add('module_one_http')
        if catalog.real and mode == 'hybrid':
            # 真实目录 + 上游可用：候选只听上游，不让合成热门路由混进来。
            routes = {item_id: {'module_one_http'} for item_id in upstream}

        # 目标商品（similar/explain/compare 需要）：不存在直接抛 ITEM_NOT_FOUND。
        targets = [catalog.get(item_id) for item_id in request.item_ids]
        if request.intent in ('explain', 'compare'):
            # 解释/对比只看用户点名的商品，不跑召回（省算力，也避免推出无关商品）。
            routes = {product['item_id']: {'requested_item'} for product in targets}
        else:
            for item_id, product in catalog.items.items():
                if catalog.real and mode == 'hybrid' and item_id not in upstream:
                    continue                                    # 真实 hybrid：非上游候选不参与
                if constraints.get('category') == product['category']:
                    routes[item_id].add('category')             # 品类命中
                if any(constraints.get(key) in product['tags'] for key in ('recipient', 'scene')):
                    routes[item_id].add('context_tags')         # 对象/场景标签命中
                if request.intent == 'similar' and targets and item_id != targets[0]['item_id']:
                    if product['category'] == targets[0]['category'] or set(product['tags']) & set(targets[0]['tags']):
                        routes[item_id].add('similarity')       # 找相似：同品类或共享标签
            if request.intent == 'similar' and targets:
                # 找相似只保留相似路由，避免“找相似”退化成“全量推荐”。
                routes = {item_id: sources for item_id, sources in routes.items() if 'similarity' in sources}

        # ③④ 逐候选：硬约束过滤 → 打分 → 证据包；被拒的进 rejected 留痕。
        rejected, items = [], []
        for item_id, sources in routes.items():
            product = catalog.items[item_id]
            failures = self._hard_failures(product, constraints)
            if failures:
                rejected.append({'item_id': item_id, 'reasons': failures})
                continue
            breakdown = self._score_breakdown(request, product, targets, upstream, mode, constraints)
            evidence = self._evidence(request, product, sources, constraints)
            items.append(RecommendationItem(
                request.trace_id, item_id, product['name'], product['price'], product['category'], product['color'],
                sorted(sources), round(sum(breakdown.values()), 6), breakdown, evidence,
                currency=product.get('currency', catalog.currency), price_kind=product.get('price_kind', 'catalog')))

        # ⑤ 稳定排序：分数降序、同分按商品 ID 升序 → 同一输入必得同一结果（阶段3 用 sha256 验证）。
        items.sort(key=lambda item: (-item.score, item.item_id))
        diagnostic = {'mode': mode, 'warnings': warnings, 'recall_count': len(routes),
                      'filtered_count': len(rejected), 'rejected': rejected,
                      'candidates': [{'item_id': key, 'sources': sorted(value)} for key, value in sorted(routes.items())]}
        return items[:request.top_k], diagnostic
