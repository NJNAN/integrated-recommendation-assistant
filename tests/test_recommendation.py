"""阶段3 单测：目录校验、硬约束、确定性 TopK、上游降级与缺失价格。"""
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
    return replace(Settings(), catalog_path=catalog_path or ROOT / 'data/catalog.json',
                   recommender_url='', recommender_username='', recommender_password='', recommender_accounts=())


def request(**kwargs):
    fields = {'trace_id': 'unit-recommend', 'user_id': 'u', 'session_id': 's',
              'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text', 'intent': 'discover'}
    fields.update(kwargs)
    return RecommendationRequest(**fields)


class TimeoutRemote:
    """上游超时桩：必须降级为离线召回，而不是把异常抛给编排层。"""

    def recall(self, request):
        raise AdapterError('UPSTREAM_TIMEOUT', '注入超时')


class CatalogTests(unittest.TestCase):
    def test_invalid_catalog_raises_contract_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text(json.dumps({'items': [{'item_id': 'x', 'name': '缺品类商品'}]}), encoding='utf-8')
            with self.assertRaises(ContractError) as caught:
                Catalog(path)
            self.assertEqual(caught.exception.code, 'INVALID_CATALOG')


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = RecommendationAdapter(demo_settings())

    def test_hard_constraints_are_not_overridden_by_scores(self):
        items, diagnostic = self.adapter.recommend(request(constraints={'category': '背包', 'budget_max': 100}))
        self.assertEqual(items, [])
        rejected = {row['item_id']: row['reasons'] for row in diagnostic['rejected']}
        self.assertEqual(rejected['demo-001'], ['budget'])       # 同类目但超预算，分数不能放行
        self.assertIn('budget', rejected['demo-002'])
        self.assertIn('category', rejected['demo-003'])          # 预算充足但品类不符

    def test_ranking_is_deterministic_and_topk_is_applied(self):
        first, diagnostic = self.adapter.recommend(request(top_k=3))
        second, _ = RecommendationAdapter(demo_settings()).recommend(request(top_k=3))
        self.assertEqual(len(first), 3)
        self.assertEqual([item.item_id for item in first], [item.item_id for item in second])
        self.assertEqual([item.score for item in first], [item.score for item in second])
        self.assertEqual([item.score for item in first], sorted([item.score for item in first], reverse=True))
        self.assertEqual(set(diagnostic), {'mode', 'warnings', 'recall_count', 'filtered_count', 'rejected', 'candidates'})

    def test_upstream_timeout_degrades_to_popularity_recall(self):
        adapter = RecommendationAdapter(demo_settings(), TimeoutRemote())
        items, diagnostic = adapter.recommend(request())
        self.assertEqual(diagnostic['mode'], 'offline_demo')
        self.assertEqual(diagnostic['warnings'][0]['code'], 'UPSTREAM_TIMEOUT')
        self.assertTrue(items)
        self.assertTrue(all('popularity' in item.sources for item in items))

    def test_missing_price_is_excluded_by_budget_filter(self):
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
            items, _ = adapter.recommend(request())
            self.assertEqual(len(items), 1)
            self.assertIsNone(items[0].price)
            self.assertEqual(items[0].price_kind, 'missing')


if __name__ == '__main__':
    unittest.main()
