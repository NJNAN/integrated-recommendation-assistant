"""阶段4 解释的单元测试：引用可核验、降级不编造、语音文本与正文一致。"""
import unittest

from assistant.contracts import EvidencePack, RecommendationItem
from assistant.explanation import explain

TRACE = 'trace-explain'


def make_item(price=199.0, attributes=None, snippets=None):
    """构造一条证据完整的推荐商品；参数可替换属性与片段用于注入异常。"""
    default_snippets = [{'id': 'catalog:demo-001:attributes', 'source': 'catalog', 'text': '目录属性'},
                        {'id': 'catalog:demo-001:description', 'source': 'catalog', 'text': '轻量双肩设计。'}]
    evidence = EvidencePack(
        trace_id=TRACE, item_id='demo-001',
        attributes=attributes or {'name': '云朵轻量双肩包', 'price': price, 'category': '背包',
                                  'color': '蓝色', 'currency': 'CNY', 'price_kind': 'catalog'},
        sources=['popularity'], matched_constraints={'category': '背包', 'budget_max': 300},
        visual_labels=[], snippets=default_snippets if snippets is None else snippets)
    return RecommendationItem(trace_id=TRACE, item_id='demo-001', name='云朵轻量双肩包', price=price,
                              category='背包', color='蓝色', sources=['popularity'], score=1.0,
                              score_breakdown={'popularity': 0.4}, evidence=evidence)


class ExplanationTest(unittest.TestCase):
    def test_citations_are_verifiable(self):
        """正常解释的每条引用都能在 evidence.snippets 里查到。"""
        item = make_item()
        result = explain(item)
        ids = {s['id'] for s in item.evidence.snippets}
        self.assertEqual(result['code'], 'OK')
        self.assertTrue(result['citations'])
        self.assertLessEqual(set(result['citations']), ids)
        self.assertTrue(all(c.startswith('catalog:demo-001:') for c in result['citations']))
        self.assertIn('¥199', result['text'])
        self.assertIn('符合当前条件', result['text'])
        self.assertFalse(result['degraded'])

    def test_no_evidence_degrades(self):
        """没有证据片段时不生成引用，也不编造解释。"""
        result = explain(make_item(snippets=[]))
        self.assertEqual(result['code'], 'INSUFFICIENT_EVIDENCE')
        self.assertTrue(result['degraded'])
        self.assertEqual(result['citations'], [])

    def test_attribute_mismatch_degrades(self):
        """证据属性缺失或与商品不一致时降级，并清空引用。"""
        attributes = {'name': '云朵轻量双肩包', 'category': '背包', 'color': '蓝色'}
        result = explain(make_item(attributes=attributes))
        self.assertEqual(result['code'], 'EVIDENCE_MISMATCH')
        self.assertEqual(result['citations'], [])
        shifted = make_item()
        shifted.evidence.attributes['price'] += 1
        self.assertEqual(explain(shifted)['code'], 'EVIDENCE_MISMATCH')

    def test_speech_text_matches_text(self):
        """TTS 直接复用正文：正常与降级都必须完全一致。"""
        normal = explain(make_item())
        self.assertEqual(normal['speech_text'], normal['text'])
        degraded = explain(make_item(snippets=[]))
        self.assertEqual(degraded['speech_text'], degraded['text'])


if __name__ == '__main__':
    unittest.main()
