"""契约层单元测试：事件必填字段与枚举、source 对应 payload、请求往返、槽位校验。"""
import unittest
from assistant.contracts import ContractError, InteractionEvent, RecommendationRequest, validate_slots


def _event(**changes):
    data = {'trace_id': 't-1', 'user_id': 'u-1', 'session_id': 's-1',
            'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text',
            'payload': {'text': '推荐礼物'}}
    data.update(changes)
    return data


class EventTests(unittest.TestCase):
    def test_valid_event_roundtrip(self):
        data = _event()
        self.assertEqual(InteractionEvent.from_dict(data).to_dict(), data)

    def test_required_metadata_and_source_enum(self):
        for field in ('trace_id', 'user_id', 'session_id', 'timestamp', 'source'):
            with self.subTest(field=field):
                data = _event()
                del data[field]
                with self.assertRaises(ContractError):
                    InteractionEvent.from_dict(data)
        with self.assertRaises(ContractError):
            InteractionEvent.from_dict(_event(source='smell'))

    def test_payload_must_match_source(self):
        cases = [{'source': 'voice', 'payload': {}},
                 {'source': 'gesture', 'payload': {'gesture': 'wave'}},
                 {'source': 'text', 'payload': {}},
                 {'source': 'vision', 'payload': {'labels': [{'label': '红色', 'confidence': 1.1}]}}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ContractError):
                InteractionEvent.from_dict(_event(**change))


class RequestTests(unittest.TestCase):
    def _request(self, **changes):
        data = {'trace_id': 't-1', 'user_id': 'u-1', 'session_id': 's-1',
                'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text', 'intent': 'filter',
                'constraints': {'category': '背包', 'budget_max': 300, 'currency': 'GBP'},
                'preferences': {'item-1': 1}, 'item_ids': ['item-1'], 'top_k': 6}
        data.update(changes)
        return data

    def test_request_roundtrip(self):
        request = RecommendationRequest.from_dict(self._request())
        self.assertEqual(RecommendationRequest.from_dict(request.to_dict()).to_dict(), request.to_dict())

    def test_invalid_request(self):
        changes = ({'intent': 'buy'}, {'top_k': 0}, {'constraints': {'unknown': 1}},
                   {'preferences': {'item-1': 2}})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ContractError) as caught:
                RecommendationRequest.from_dict(self._request(**change))
            self.assertEqual(caught.exception.code, 'INVALID_REQUEST')


class SlotTests(unittest.TestCase):
    def test_valid_slots_pass_through(self):
        slots = {'category': '背包', 'budget_min': 100, 'budget_max': 300, 'currency': 'GBP'}
        self.assertIs(validate_slots(slots), slots)

    def test_invalid_slots_rejected(self):
        for slots in ({'unknown': 1}, {'currency': 'USD'}, {'budget_max': -1}, []):
            with self.subTest(slots=slots), self.assertRaises(ContractError):
                validate_slots(slots)


if __name__ == '__main__':
    unittest.main()
