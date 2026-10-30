"""多路召回 + 硬约束过滤 + 确定性重排：阶段3 的核心模块。"""
import json
from pathlib import Path
from assistant.adapters import AdapterError, HTTPRecommender
from assistant.contracts import CURRENCIES, ContractError, EvidencePack, RecommendationItem, number, nonempty


class Catalog:
    """读取并校验商品目录；任何脏数据都抛 INVALID_CATALOG，绝不半信半疑地使用。"""

    def __init__(self, path):
        try:
            document = json.loads(Path(path).read_text(encoding='utf-8'))
            rows = document['items']
            if not isinstance(rows, list) or not rows:
                raise ValueError('商品目录为空')
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
                for key in ('item_id', 'name', 'category', 'color'):
                    nonempty(row.get(key), key)
                if row.get('price') is None:
                    if not self.real or row.get('price_kind') != 'missing':
                        raise ValueError(f"商品 {row.get('item_id')} 缺失价格必须来自真实目录并标注 missing")
                else:
                    number(row['price'], 'price')
                number(row.get('popularity'), 'popularity', 0, 1)
                if not isinstance(row.get('tags', []), list) or any(
                        not isinstance(tag, str) for tag in row.get('tags', [])):
                    raise ValueError('tags 必须是字符串数组')
                if not isinstance(row.get('description', ''), str):
                    raise ValueError('description 必须是字符串')
                for label in row.get('visual_labels', []):
                    nonempty(label.get('label'), 'label')
                    number(label.get('confidence'), 'confidence', 0, 1)
                if row['item_id'] in self.items:
                    raise ValueError(f"商品ID重复：{row['item_id']}")
                row['tags'] = list(row.get('tags', []))
                row['description'] = row.get('description', '')
                row['visual_labels'] = list(row.get('visual_labels', []))
                self.items[row['item_id']] = row
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ContractError(f'商品目录读取或校验失败：{exc}', 'INVALID_CATALOG') from exc

    def get(self, item_id):
        """按 ID 取商品；缺失时抛 ITEM_NOT_FOUND，调用方据此返回明确错误。"""
        if item_id not in self.items:
            raise ContractError(f'商品不存在：{item_id}', 'ITEM_NOT_FOUND')
        return self.items[item_id]


class RecommendationAdapter:
    """召回 → 过滤 → 打分 → 组装证据包；上游任何失败都降级为离线召回。"""

    def __init__(self, settings, remote=None):
        self.settings = settings
        self.catalog = Catalog(settings.catalog_path)
        self.remote = remote if remote is not None else (HTTPRecommender(settings) if settings.recommender_url else None)

    def _recall_upstream(self, request):
        """调上游只取候选与分数；超时/失联/格式错误都降级，并保留错误码。"""
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
                if row['item_id'] in self.catalog.items:
                    scores[row['item_id']] = max(score, scores.get(row['item_id'], -1000))
            if rows and not scores:
                raise ContractError('上游商品ID与配置目录不匹配，不能编造价格或属性', 'INVALID_UPSTREAM')
            return scores, [], 'hybrid'
        except (AdapterError, ContractError, OSError, TimeoutError) as exc:
            code = 'INVALID_UPSTREAM' if isinstance(exc, ContractError) else getattr(exc, 'code', 'UPSTREAM_UNAVAILABLE')
            return {}, [{'code': code, 'message': str(exc)}], 'offline_demo'

    @staticmethod
    def _hard_failures(product, constraints):
        """硬约束在打分前判定：分数再高也不能覆盖品类、颜色、预算与缺失价格。"""
        failures = [key for key in ('category', 'color') if key in constraints and product[key] != constraints[key]]
        price = product['price']
        if price is None and any(key in constraints for key in ('budget_min', 'budget_max')):
            failures.append('missing_price')
        elif price is not None and (price < constraints.get('budget_min', 0)
                                    or price > constraints.get('budget_max', float('inf'))):
            failures.append('budget')
        return failures

    def _score_breakdown(self, request, product, targets, upstream, mode, constraints):
        """SPEC 第 4 节的六项权重；阶段5 的排名变化完全依赖这里的数值。"""
        tags = product['tags']
        popularity = 0 if self.catalog.real and mode == 'hybrid' else .4 * product['popularity']
        context = .2 * sum(constraints.get(key) in tags for key in ('recipient', 'scene'))
        similarity = 0
        if request.intent == 'similar' and targets:
            similarity = .3 * (product['category'] == targets[0]['category'])
            similarity += .1 * len(set(tags) & set(targets[0]['tags']))
        upstream_weight = 1 if self.catalog.real else .3
        affinity = sum(sign * .06 * len(set(self.catalog.items[liked]['tags']) & set(tags))
                       for liked, sign in request.preferences.items()
                       if liked in self.catalog.items and liked != product['item_id'])
        return {'popularity': round(popularity, 6), 'context': round(context, 6),
                'similarity': round(similarity, 6), 'upstream': round(upstream_weight * upstream.get(product['item_id'], 0), 6),
                'feedback': round(2.5 * request.preferences.get(product['item_id'], 0), 6),
                'tag_affinity': round(affinity, 6)}

    def _evidence(self, request, product, sources, constraints):
        """证据包只放目录事实；解释模块只允许引用这些片段。"""
        item_id = product['item_id']
        attributes = {'name': product['name'], 'price': product['price'], 'category': product['category'],
                      'color': product['color'], 'currency': product.get('currency', self.catalog.currency),
                      'price_kind': product.get('price_kind', 'catalog')}
        snippets = [{'id': f'catalog:{item_id}:attributes', 'source': 'catalog',
                     'text': json.dumps(attributes, ensure_ascii=False)}]
        if product['description']:
            snippets.append({'id': f'catalog:{item_id}:description', 'source': 'catalog', 'text': product['description']})
        if self.catalog.real:
            provenance = {key: product.get(key) for key in ('price_source', 'color_source', 'original_category')}
            snippets.append({'id': f'catalog:{item_id}:provenance', 'source': 'module_one_catalog',
                             'text': json.dumps(provenance, ensure_ascii=False)})
        labels = [tag for tag in product['visual_labels'] if tag['confidence'] >= self.settings.visual_threshold]
        matched = {key: value for key, value in constraints.items()
                   if key in ('category', 'color', 'budget_min', 'budget_max') or value in product['tags']}
        return EvidencePack(request.trace_id, item_id, attributes, sorted(sources), matched, labels, snippets)

    def recommend(self, request):
        """返回 (按分数降序的 TopK 商品, diagnostic)；同分按 item_id 保证可复现。"""
        catalog, constraints = self.catalog, request.constraints
        upstream, warnings, mode = self._recall_upstream(request)
        routes = {item_id: {'popularity'} for item_id in catalog.items}
        for item_id in upstream:
            routes[item_id].add('module_one_http')
        if catalog.real and mode == 'hybrid':
            routes = {item_id: {'module_one_http'} for item_id in upstream}
        targets = [catalog.get(item_id) for item_id in request.item_ids]
        if request.intent in ('explain', 'compare'):
            routes = {product['item_id']: {'requested_item'} for product in targets}
        else:
            for item_id, product in catalog.items.items():
                if catalog.real and mode == 'hybrid' and item_id not in upstream:
                    continue
                if constraints.get('category') == product['category']:
                    routes[item_id].add('category')
                if any(constraints.get(key) in product['tags'] for key in ('recipient', 'scene')):
                    routes[item_id].add('context_tags')
                if request.intent == 'similar' and targets and item_id != targets[0]['item_id']:
                    if product['category'] == targets[0]['category'] or set(product['tags']) & set(targets[0]['tags']):
                        routes[item_id].add('similarity')
            if request.intent == 'similar' and targets:
                routes = {item_id: sources for item_id, sources in routes.items() if 'similarity' in sources}

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
        items.sort(key=lambda item: (-item.score, item.item_id))
        diagnostic = {'mode': mode, 'warnings': warnings, 'recall_count': len(routes),
                      'filtered_count': len(rejected), 'rejected': rejected,
                      'candidates': [{'item_id': key, 'sources': sorted(value)} for key, value in sorted(routes.items())]}
        return items[:request.top_k], diagnostic
