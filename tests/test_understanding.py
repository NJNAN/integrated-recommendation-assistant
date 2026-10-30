"""意图/槽位/会话的单元测试：只看 understanding.py、session.py、contracts.py。"""
from types import SimpleNamespace
import unittest
from assistant.contracts import InteractionEvent
from assistant.session import SessionContext, SessionStore
from assistant.understanding import understand


def event(payload, source='text'):
    """构造一条合法事件；时间戳固定，保证测试结果可复现。"""
    return InteractionEvent.from_dict({'trace_id': 'unit', 'user_id': 'u', 'session_id': 's',
                                       'timestamp': '2026-10-30T10:00:00+08:00',
                                       'source': source, 'payload': payload})


def slots_of(text, source='text', key='text'):
    return understand(event({key: text}, source)).slots


class UnderstandingTests(unittest.TestCase):
    def test_six_intents_and_priority(self):
        cases = [(event({'text': '推荐礼物'}), 'discover'), (event({'text': '蓝色背包'}), 'filter'),
                 (event({'intent': 'similar', 'item_id': 'a'}), 'similar'),
                 (event({'intent': 'compare', 'item_ids': ['a', 'b']}), 'compare'),
                 (event({'intent': 'explain', 'item_id': 'a'}), 'explain'),
                 (event({'text': '为什么推荐这个', 'item_id': 'a'}), 'explain'),
                 (event({'text': '不喜欢'}), 'feedback'), (event({'text': '喜欢蓝色背包'}), 'feedback')]
        for item, intent in cases:
            with self.subTest(intent=intent, text=item.payload):
                self.assertEqual(understand(item).intent, intent)

    def test_category_color_and_context_slots(self):
        result = understand(event({'text': '给朋友推荐蓝色双肩包'}))
        self.assertEqual(result.slots, {'category': '背包', 'color': '蓝色', 'recipient': '朋友'})
        self.assertEqual(result.intent, 'filter')
        self.assertEqual(slots_of('学习用的台灯'), {'category': '台灯', 'scene': '学习'})
        for alias, canonical in (('保温杯', '杯子'), ('晴雨伞', '雨伞')):
            self.assertEqual(slots_of(alias)['category'], canonical)

    def test_budget_bounds_range_and_currency(self):
        self.assertEqual(slots_of('预算300元以内')['budget_max'], 300.0)
        self.assertEqual(slots_of('至少100元')['budget_min'], 100.0)
        self.assertEqual(slots_of('预算100到300元'), {'budget_min': 100.0, 'budget_max': 300.0})
        self.assertEqual(slots_of('预算1,000元以内的耳机')['budget_max'], 1000.0)
        self.assertEqual(understand(event({'text': '预算10英镑以内'})).slots['currency'], 'GBP')

    def test_unparsed_budget_asks_with_fixed_message(self):
        result = understand(event({'text': '耳机预算三百元'}))
        self.assertNotIn('budget_max', result.slots)
        self.assertIn('预算金额（请使用阿拉伯数字，例如300元以内）', result.missing)
        self.assertNotIn('budget_max', understand(event({'text': '预算1,00元以内'})).slots)

    def test_same_round_conflict_keeps_first_value(self):
        result = understand(event({'text': '蓝色', 'labels': [{'label': '红色', 'confidence': .99}]}, 'vision'))
        self.assertEqual(result.slots['color'], '蓝色')
        self.assertEqual(result.conflicts[0]['slot'], 'color')

    def test_low_confidence_labels_are_ignored_not_slotted(self):
        result = understand(event({'labels': [{'label': '背包', 'confidence': .9},
                                              {'label': '红色', 'confidence': .3}]}, 'vision'))
        self.assertEqual(result.slots, {'category': '背包'})
        self.assertEqual(result.ignored_labels, [{'label': '红色', 'confidence': .3}])
        self.assertIn('可信视觉标签或补充文字',
                      understand(event({'labels': [{'label': '红色', 'confidence': .3}]}, 'vision')).missing)

    def test_gesture_feedback_and_actions(self):
        def result(gesture, item_id='a', text=None):
            payload = {'gesture': gesture, 'item_id': item_id}
            if text:
                payload['text'] = text
            return understand(event(payload, 'gesture'))
        self.assertEqual((result('thumbs_up').intent, result('thumbs_up').feedback), ('feedback', 1))
        self.assertEqual((result('thumbs_down').intent, result('thumbs_down').feedback), ('feedback', -1))
        self.assertEqual(result('open_palm').action, 'end')
        self.assertEqual(result('victory').intent, 'discover')
        self.assertIn('item_id', understand(event({'gesture': 'thumbs_up'}, 'gesture')).missing)


class SessionTests(unittest.TestCase):
    def test_merge_conflict_does_not_pollute_and_undo_restores(self):
        ctx = SessionContext('u', 's')
        self.assertEqual(ctx.merge({'category': '背包'}, {'category': {'source': 'text'}}), [])
        ctx.merge({'budget_max': 300}, {'budget_max': {'source': 'text'}})
        conflicts = ctx.merge({'budget_min': 500}, {'budget_min': {'source': 'text'}})
        self.assertEqual(conflicts[0]['slot'], 'budget')
        self.assertEqual(ctx.constraints, {'category': '背包', 'budget_max': 300})
        self.assertEqual(len(ctx.history), 2)          # 冲突那轮没有压栈
        ctx.undo()
        self.assertEqual(ctx.constraints, {'category': '背包'})

    def test_store_ttl_and_end(self):
        clock = SimpleNamespace(now=0.0)
        settings = SimpleNamespace(session_ttl_seconds=10, max_sessions=2, top_k=6)
        store = SessionStore(settings, clock=lambda: clock.now)
        first = store.get('u', 'a')
        first.merge({'color': '蓝色'}, {})
        clock.now = 11
        self.assertEqual(store.get('u', 'a').constraints, {})     # 过期后重建空会话
        session = store.get('u', 'b')
        store.end('u', 'b')
        self.assertIsNot(store.get('u', 'b'), session)            # 结束后是新会话


if __name__ == '__main__':
    unittest.main()
