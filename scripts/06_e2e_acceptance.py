"""阶段06：端到端验收——语音/视觉/手势链路、故障降级、非法输入、会话结束与边界空结果。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stage import case, demo_settings, event, finish, report, write_json  # noqa: E402
from assistant.adapters import AdapterError  # noqa: E402
from assistant.orchestrator import Orchestrator  # noqa: E402
from assistant.recommendation import RecommendationAdapter  # noqa: E402


class TimeoutRemote:
    """假推荐上游：每次召回都抛超时，用来验证降级路径。
    例如注入后 06 仍应返回 popularity 召回结果，而不是空列表。
    """

    def recall(self, request):
        """固定抛 UPSTREAM_TIMEOUT，模拟模块一挂掉或网络超时。
        例如 fault 用例拿到的 warnings 里就是这条错误码。
        """
        raise AdapterError('UPSTREAM_TIMEOUT', '上游推荐超时')


def _traced(response):
    """检查 trace 与商品是否共用同一 trace_id。
    例如响应 trace_id 是 e2e-voice，items 里混进别的 id 就算失败。
    """
    return (bool(response['trace'])
            and all(entry['trace_id'] == response['trace_id'] for entry in response['trace'])
            and all(item['trace_id'] == response['trace_id'] for item in response['items']))


def run(settings):
    """跑 7 条端到端用例，覆盖三条模态链路、降级、非法输入与空集。
    例如三条链路都要断言同一 trace_id，结果落盘 JSON。
    """
    app = Orchestrator(settings)
    voice = case('语音转写链路', app, event('e2e-voice', source='voice',
                 payload={'transcript': '给朋友买礼物，预算300以内'}, session='e2e-voice'), 'ok',
                 lambda r: (_traced(r)
                            and r['context']['constraints'] == {'recipient': '朋友', 'budget_max': 300}
                            and bool(r['items'])
                            and all(i['price'] <= 300 and i['sources'] for i in r['items'])))
    vision = case('视觉标签链路', app, event('e2e-vision', source='vision',
                  payload={'labels': [{'label': '背包', 'confidence': .95}, {'label': '蓝色', 'confidence': .9}]},
                  session='e2e-vision'), 'ok',
                  lambda r: (_traced(r) and r['context']['constraints'] == {'category': '背包', 'color': '蓝色'}
                             and bool(r['items'])
                             and all(i['category'] == '背包' and i['color'] == '蓝色' for i in r['items'])))
    target = vision['response']['items'][0]['item_id']
    gesture = case('手势反馈链路', app, event('e2e-gesture', source='gesture',
                  payload={'gesture': 'thumbs_up', 'item_id': target}, session='e2e-vision'), 'ok',
                  lambda r: (_traced(r) and r['context']['preferences'].get(target) == 1
                             and target in [i['item_id'] for i in r['items']]))
    faulty = Orchestrator(settings, RecommendationAdapter(settings, TimeoutRemote()))
    fault = case('故障降级链路', faulty, event('e2e-fault', '推荐耳机预算300以内'), 'ok',
                 lambda r: (_traced(r) and r['degraded'] and r['warnings'][0]['code'] == 'UPSTREAM_TIMEOUT'
                            and bool(r['items'])
                            and all(i['category'] == '耳机' and i['price'] <= 300
                                    and 'popularity' in i['sources'] for i in r['items'])))
    invalid = case('非法输入链路', app, {'payload': {}}, 'error',
                   lambda r: r['error']['code'] == 'INVALID_EVENT' and _traced(r))
    ended = case('会话结束', app, event('e2e-end', source='gesture', payload={'gesture': 'open_palm'}), 'ended',
                 lambda r: r['status'] == 'ended' and r['context'] == {} and _traced(r))
    empty = case('边界空结果', app, event('e2e-empty', '推荐耳机预算1元以内'), 'empty',
                 lambda r: (_traced(r) and r['items'] == []
                            and r['context']['constraints'] == {'category': '耳机', 'budget_max': 1}
                            and '调整' in r['message']))
    cases = [voice, vision, gesture, fault, invalid, ended, empty]
    output = report(cases)
    output['scope'] = {
        'voice': '已转写的 ASR 文本（音频链路由模块二真实联调另行记录）',
        'vision': '已识别的视觉标签（置信度由模型自报）',
        'gesture': '已确认的手势事件',
        'fault': '适配器故障注入：UPSTREAM_TIMEOUT，验证降级而不中断服务',
        'boundary': '耳机预算1元以内触发硬约束空集，不放宽预算',
        'trace_id': '每条响应的 trace 与 items 使用同一 trace_id',
        'real_models_verified': False,
    }
    write_json(settings, 'stage06_acceptance.json', output)
    finish(settings, cases, '阶段06')
    return cases


if __name__ == '__main__':
    run(demo_settings())
