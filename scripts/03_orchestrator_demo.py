"""阶段3 验收：正常推荐、硬约束空集、上游超时降级、确定性复现。"""
import hashlib
import json
from _stage import case, demo_settings, event, finish, report, write_json
from assistant.adapters import AdapterError
from assistant.orchestrator import Orchestrator
from assistant.recommendation import RecommendationAdapter


class TimeoutRemote:
    """验收用故障注入桩：上游必超时，验证降级后仍能离线召回。"""

    def recall(self, request):
        raise AdapterError('UPSTREAM_TIMEOUT', '验收故障注入：上游超时，启用离线召回')


def digest(response):
    """只摘录可复现的 items 与 context，避免耗时字段干扰比对。"""
    canonical = {'items': response['items'], 'context': response['context']}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def run(settings):
    app = Orchestrator(settings)
    degraded_app = Orchestrator(settings, RecommendationAdapter(settings, TimeoutRemote()))

    def normal_ok(response):
        scores = [item['score'] for item in response['items']]
        return (response['status'] == 'ok' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all(item['price'] is not None and item['price'] <= 300 for item in response['items'])
                and all(item['sources'] for item in response['items'])
                and scores == sorted(scores, reverse=True))

    def empty_ok(response):
        return (response['items'] == [] and '调整' in response['message']
                and response['context']['constraints'] == {'category': '耳机', 'budget_max': 1})

    def degraded_ok(response):
        return (response['status'] == 'ok' and response['degraded']
                and response['warnings'][0]['code'] == 'UPSTREAM_TIMEOUT' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all('popularity' in item['sources'] for item in response['items']))

    cases = [case('正常推荐', app, event('orchestrator-normal', '蓝色背包预算300以内'), 'ok', normal_ok),
             case('硬约束空集', app, event('orchestrator-empty', '耳机预算1以内', session='empty'), 'empty', empty_ok),
             case('上游超时降级', degraded_app, event('orchestrator-timeout', '蓝色背包预算300以内'), 'ok', degraded_ok)]

    repro_event = event('repro', '推荐礼物预算300以内')
    first, second = Orchestrator(settings).handle(repro_event), Orchestrator(settings).handle(repro_event)
    first_digest, second_digest = digest(first), digest(second)
    same = (first_digest == second_digest and first['items'] == second['items']
            and first['context'] == second['context'])
    cases.append({'name': '确定性复现', 'input': repro_event, 'expected_status': first['status'],
                  'response': first, 'second_response': second, 'passed': bool(same),
                  'sha256': {'first': first_digest, 'second': second_digest, 'equal': first_digest == second_digest},
                  'comparison': {'same_items_and_scores': first['items'] == second['items'],
                                 'same_context': first['context'] == second['context']}})
    write_json(settings, 'stage03_demo.json', report(cases))
    finish(settings, cases, '阶段03')
    return cases


if __name__ == '__main__':
    run(demo_settings())
