"""阶段3 单测：目录校验、硬约束优先级、确定性 TopK、上游降级、缺失价格。

跑法：`python -B -m unittest tests.test_recommendation -v`（5 条用例）
每个用例都直接打 `RecommendationAdapter`，不依赖 HTTP/会话，失败时能一眼定位到过滤或打分。
"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from assistant.adapters import AdapterError
from assistant.contracts import ContractError, RecommendationRequest
from assistant.recommendation import Catalog, RecommendationAdapter
from config.settings import ROOT, Settings


def demo_settings(catalog_path=None):
    """离线设置：固定合成目录、清空上游配置，保证单测不依赖网络和本机凭据。"""
    return replace(Settings(), catalog_path=catalog_path or ROOT / 'data/catalog.json',
                   recommender_url='', recommender_username='', recommender_password='', recommender_accounts=())


def request(**kwargs):
    """构造一条合法的推荐请求，用例里只覆盖关心的字段。"""
    fields = {'trace_id': 'unit-recommend', 'user_id': 'u', 'session_id': 's',
              'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text', 'intent': 'discover'}
    fields.update(kwargs)
    return RecommendationRequest(**fields)


class TimeoutRemote:
    """上游超时桩：必须降级为离线召回，而不是把异常抛给编排层。"""

    def recall(self, request):
        raise AdapterError('UPSTREAM_TIMEOUT', '注入超时')


class CatalogTests(unittest.TestCase):
    """目录校验：脏数据必须在加载阶段失败，并且错误码明确。"""

    def test_invalid_catalog_raises_contract_error(self):
        # 缺 category/color/price 的商品不允许被加载，否则后面过滤和解释都会拿到 None。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text(json.dumps({'items': [{'item_id': 'x', 'name': '缺品类商品'}]}), encoding='utf-8')
            with self.assertRaises(ContractError) as caught:
                Catalog(path)
            self.assertEqual(caught.exception.code, 'INVALID_CATALOG')


class RecommendationTests(unittest.TestCase):
    """推荐主流程：硬约束、排序确定性、降级、缺失价格。"""

    def setUp(self):
        self.adapter = RecommendationAdapter(demo_settings())

    def test_hard_constraints_are_not_overridden_by_scores(self):
        # 背包 + 预算 100：demo-001 品类对但超预算 → 必须被拒，分数再高也不行。
        items, diagnostic = self.adapter.recommend(request(constraints={'category': '背包', 'budget_max': 100}))
        self.assertEqual(items, [])
        rejected = {row['item_id']: row['reasons'] for row in diagnostic['rejected']}
        self.assertEqual(rejected['demo-001'], ['budget'])       # 同类目但超预算，分数不能放行
        self.assertIn('budget', rejected['demo-002'])
        self.assertIn('category', rejected['demo-003'])          # 预算充足但品类不符

    def test_ranking_is_deterministic_and_topk_is_applied(self):
        # 同一请求跑两次（两个独立 adapter），item_id / score 顺序必须完全一致。
        first, diagnostic = self.adapter.recommend(request(top_k=3))
        second, _ = RecommendationAdapter(demo_settings()).recommend(request(top_k=3))
        self.assertEqual(len(first), 3)                          # TopK 截断生效
        self.assertEqual([item.item_id for item in first], [item.item_id for item in second])
        self.assertEqual([item.score for item in first], [item.score for item in second])
        self.assertEqual([item.score for item in first], sorted([item.score for item in first], reverse=True))
        # diagnostic 的键集合是稳定契约，阶段3 的 trace 和前端都依赖它。
        self.assertEqual(set(diagnostic), {'mode', 'warnings', 'recall_count', 'filtered_count', 'rejected', 'candidates'})

    def test_upstream_timeout_degrades_to_popularity_recall(self):
        # 注入必超时的上游：模式变 offline_demo、warnings 带错误码、结果仍来自 popularity 路由。
        adapter = RecommendationAdapter(demo_settings(), TimeoutRemote())
        items, diagnostic = adapter.recommend(request())
        self.assertEqual(diagnostic['mode'], 'offline_demo')
        self.assertEqual(diagnostic['warnings'][0]['code'], 'UPSTREAM_TIMEOUT')
        self.assertTrue(items)
        self.assertTrue(all('popularity' in item.sources for item in items))

    def test_missing_price_is_excluded_by_budget_filter(self):
        # 真实目录允许缺价商品：不带预算时可以展示（标注 missing），带预算时必须被 missing_price 排除。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'catalog.json'
            path.write_text(json.dumps({'dataset': 'unit', 'currency': 'CNY', 'source_kind': 'module_one_real',
                                        'items': [{'item_id': 'no-price', 'name': '无价商品', 'category': '杯子',
                                                   'color': '白色', 'price': None, 'price_kind': 'missing',
                                                   'popularity': 0.5, 'tags': [], 'description': '价格缺失'}]},
                                       ensure_ascii=False), encoding='utf-8')
            adapter = RecommendationAdapter(demo_settings(path))
            items, diagnostic = adapter.recommend(request(constraints={'budget_max': 100}))
            self.assertEqual(items, [])
            self.assertEqual(diagnostic['rejected'][0]['reasons'], ['missing_price'])
            items, _ = adapter.recommend(request())              # 不给预算：可以展示，但明确标记缺失
            self.assertEqual(len(items), 1)
            self.assertIsNone(items[0].price)
            self.assertEqual(items[0].price_kind, 'missing')


if __name__ == '__main__':
    unittest.main()
