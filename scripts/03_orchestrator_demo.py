"""阶段3 验收（可重跑）：正常推荐、硬约束空集、上游超时降级、确定性复现。

跑法：`python -B scripts/03_orchestrator_demo.py`
产物：`outputs/stage03_demo.json`（每条用例含 input / expected_status / response / passed）
约定：4 条用例必须全过；任何一条失败，`finish()` 会以非零状态退出。
"""
import hashlib
import json
from _stage import case, demo_settings, event, finish, report, write_json
from assistant.adapters import AdapterError
from assistant.orchestrator import Orchestrator
from assistant.recommendation import RecommendationAdapter


class TimeoutRemote:
    """故障注入桩：任何召回都抛 UPSTREAM_TIMEOUT。

    它是 RecommendationAdapter 的 remote 参数，用来模拟“模块一挂了”；
    正确行为是降级成离线热门召回，而不是把异常抛给调用方。
    """

    def recall(self, request):
        raise AdapterError('UPSTREAM_TIMEOUT', '验收故障注入：上游超时，启用离线召回')


def digest(response):
    """对可复现部分做 sha256：只摘 items 和 context，排除 elapsed_ms 这类时间字段。"""
    canonical = {'items': response['items'], 'context': response['context']}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def run(settings):
    # 正常链路用离线合成目录；降级链路额外注入一个必超时的假上游。
    app = Orchestrator(settings)
    degraded_app = Orchestrator(settings, RecommendationAdapter(settings, TimeoutRemote()))

    def normal_ok(response):
        """正常推荐：品类/颜色/预算全符合，有来源可追溯，分数降序。"""
        scores = [item['score'] for item in response['items']]
        return (response['status'] == 'ok' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all(item['price'] is not None and item['price'] <= 300 for item in response['items'])
                and all(item['sources'] for item in response['items'])
                and scores == sorted(scores, reverse=True))

    def empty_ok(response):
        """硬约束空集：没有商品，但消息要提示“调整”，且会话约束原样保留（不放宽预算）。"""
        return (response['items'] == [] and '调整' in response['message']
                and response['context']['constraints'] == {'category': '耳机', 'budget_max': 1})

    def degraded_ok(response):
        """上游超时降级：degraded + 明确的错误码，仍然按 popularity 离线召回且硬约束生效。"""
        return (response['status'] == 'ok' and response['degraded']
                and response['warnings'][0]['code'] == 'UPSTREAM_TIMEOUT' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all('popularity' in item['sources'] for item in response['items']))

    # 前三条：分别验证成功、边界（空集）、失败降级三类行为。
    cases = [case('正常推荐', app, event('orchestrator-normal', '蓝色背包预算300以内'), 'ok', normal_ok),
             case('硬约束空集', app, event('orchestrator-empty', '耳机预算1以内', session='empty'), 'empty', empty_ok),
             case('上游超时降级', degraded_app, event('orchestrator-timeout', '蓝色背包预算300以内'), 'ok', degraded_ok)]

    # 第四条：同一事件用两个全新 Orchestrator 各跑一次，比较 items+context 的 sha256。
    # 排序键是 (-score, item_id) 且没有随机数，所以两次结果必须逐字段一致。
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
    # 落盘 + 打印；finish 在有失败时抛 SystemExit(1)，保证“跑失败就是失败”。
    write_json(settings, 'stage03_demo.json', report(cases))
    finish(settings, cases, '阶段03')
    return cases


if __name__ == '__main__':
    run(demo_settings())
