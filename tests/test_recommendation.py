"""阶段3 单测：目录校验、硬约束优先级、确定性 TopK、上游降级、缺失价格。

跑法：`python -B -m unittest tests.test_recommendation -v`（5 条用例）
每个用例都直接打 `RecommendationAdapter`，不依赖 HTTP/会话，失败时能一眼定位到过滤或打分。

阅读约定：每个函数的 docstring 都写了「参数（每个参数是什么、从哪来、取值）」「返回」「验证点」。
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
    """造一份“离线测试配置”：固定商品目录、清空所有上游/账号信息。

    参数：
        catalog_path (str | pathlib.Path | None): 商品目录 JSON 的路径。
            - 传 None：用仓库里的 18 件合成目录 `data/catalog.json`；
            - 传路径：用指定的目录文件（缺价格那个用例就是自己造一份临时目录）。

    返回：
        Settings: 一份冻结的配置副本，其中
            `catalog_path` 用上面说的目录，`recommender_url`/`recommender_username`/
            `recommender_password`/`recommender_accounts` 全部清空。
            这样测试不会去连真实服务，也不会受本机环境变量影响。

    说明：
        用 `dataclasses.replace` 复制而不是改全局 `settings`，避免测试之间互相污染。
    """
    return replace(Settings(), catalog_path=catalog_path or ROOT / 'data/catalog.json',
                   recommender_url='', recommender_username='', recommender_password='', recommender_accounts=())


def request(**kwargs):
    """造一条“合法的最小推荐请求”，用例只覆盖自己关心的字段。

    参数：
        **kwargs: 想在默认值基础上覆盖的字段，键名同合同 `RecommendationRequest`：
            - intent (str)：意图，默认 'discover'；找相似时传 'similar'；
            - constraints (dict)：约束，例如 {'category': '背包', 'budget_max': 100}；
            - preferences (dict)：反馈，例如 {'demo-001': 1}；
            - item_ids (list[str])：点名的商品，similar/explain/compare 用；
            - top_k (int)：要几件，默认 6，用例里传 3 验证截断。

    返回：
        RecommendationRequest: 校验通过的请求对象。
            默认值：trace_id='unit-recommend'、user_id='u'、session_id='s'、
            timestamp='2026-10-08T10:00:00+08:00'、source='text'、intent='discover'。
            这些是契约要求的必填字段，固定下来让测试可重复。

    异常：
        ContractError: 覆盖的字段非法时（例如 intent 不在六种里、top_k 越界），
            由 RecommendationRequest 的校验抛出。
    """
    fields = {'trace_id': 'unit-recommend', 'user_id': 'u', 'session_id': 's',
              'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text', 'intent': 'discover'}
    fields.update(kwargs)
    return RecommendationRequest(**fields)


class TimeoutRemote:
    """上游超时桩：必须降级为离线召回，而不是把异常抛给编排层。"""

    def recall(self, request):
        """假装上游超时。

        参数：
            request (RecommendationRequest): 适配器传进来的请求；这个桩不看它。

        返回：
            不会返回。

        异常：
            AdapterError(code='UPSTREAM_TIMEOUT')：和真实超时同一个错误码，
            用来验证适配器会走降级分支。
        """
        raise AdapterError('UPSTREAM_TIMEOUT', '注入超时')


class CatalogTests(unittest.TestCase):
    """目录校验：脏数据必须在加载阶段失败，并且错误码明确。"""

    def test_invalid_catalog_raises_contract_error(self):
        """验证：商品目录缺字段时，构造 Catalog 直接报 INVALID_CATALOG。

        构造：临时写一个只有 item_id/name、缺 category/color/price 的商品。
        断言：`Catalog(path)` 抛 ContractError，且 `code == 'INVALID_CATALOG'`。
        为什么：缺字段的商品如果被放进来，后面的过滤和解释会拿到 None，
        错误会以更难查的形式出现在别处，所以要在加载时就把脏数据挡住。
        """
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
        """每条用例前都新建一个适配器（用默认合成目录），保证用例之间互不影响。"""
        self.adapter = RecommendationAdapter(demo_settings())

    def test_hard_constraints_are_not_overridden_by_scores(self):
        """验证：硬约束先于打分——超预算的商品，分数再高也不能被推出来。

        构造：约束「品类=背包、预算≤100」，合成目录里没有 100 元以下的背包。
        断言：
            - 返回商品为空列表；
            - `rejected` 里 demo-001（229 元背包）的原因是 `['budget']`；
            - demo-002（黑色背包）含 `budget`；demo-003（杯子）含 `category`。
        为什么：如果实现写反了（先打分后过滤），就会返回超预算的商品，这条会红。
        """
        # 背包 + 预算 100：demo-001 品类对但超预算 → 必须被拒，分数再高也不行。
        items, diagnostic = self.adapter.recommend(request(constraints={'category': '背包', 'budget_max': 100}))
        self.assertEqual(items, [])
        rejected = {row['item_id']: row['reasons'] for row in diagnostic['rejected']}
        self.assertEqual(rejected['demo-001'], ['budget'])       # 同类目但超预算，分数不能放行
        self.assertIn('budget', rejected['demo-002'])
        self.assertIn('category', rejected['demo-003'])          # 预算充足但品类不符

    def test_ranking_is_deterministic_and_topk_is_applied(self):
        """验证：TopK 截断生效 + 同样请求两次结果完全一致 + diagnostic 字段稳定。

        构造：不带约束要前 3 件，跑两个独立的适配器各算一次。
        断言：
            - 返回正好 3 件（截断）；
            - 两次的 item_id 顺序、分数完全一致；
            - 分数是降序的；
            - diagnostic 的键集合固定为 6 个（下游 trace/前端依赖这个契约）。
        为什么：这是验收清单“同输入和配置可复现”的单测版本。
        """
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
        """验证：上游超时后降级成离线热门召回，而不是报错或返回空。

        构造：给适配器注入 `TimeoutRemote`（一调 up 就抛 UPSTREAM_TIMEOUT）。
        断言：
            - `diagnostic['mode'] == 'offline_demo'`（模式切到离线）；
            - `warnings[0]['code'] == 'UPSTREAM_TIMEOUT'`（原因码正确）；
            - 仍然有商品，且每件商品的 sources 都含 `popularity`（走的是热门兜底）。
        为什么：这是验收清单“降级”的单测版本；上游挂了用户仍要能拿到结果。
        """
        # 注入必超时的上游：模式变 offline_demo、warnings 带错误码、结果仍来自 popularity 路由。
        adapter = RecommendationAdapter(demo_settings(), TimeoutRemote())
        items, diagnostic = adapter.recommend(request())
        self.assertEqual(diagnostic['mode'], 'offline_demo')
        self.assertEqual(diagnostic['warnings'][0]['code'], 'UPSTREAM_TIMEOUT')
        self.assertTrue(items)
        self.assertTrue(all('popularity' in item.sources for item in items))

    def test_missing_price_is_excluded_by_budget_filter(self):
        """验证：价格缺失的商品——不带预算能展示（并标记缺失），带预算必须被排除。

        构造：临时造一份“真实目录风格”的目录，里面只有一件 price=None、
            price_kind='missing' 的商品（只有真实目录允许缺价）。
        断言：
            - 带 `budget_max=100` 时：返回空，且 rejected 原因是 `['missing_price']`；
            - 不带预算时：能返回 1 件，`price is None`、`price_kind == 'missing'`。
        为什么：没有价格就没有预算依据；但完全不展示也会浪费商品，所以两种行为都要验证。
        """
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
