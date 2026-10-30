"""阶段1 契约验收：事件校验、边界预算、空筛选追问与推荐请求 JSON 往返。"""
import sys
from _stage import case, demo_settings, event, finish, report, write_json
from assistant.contracts import RecommendationRequest
from assistant.orchestrator import Orchestrator


def _event_cases(app):
    """6 条契约用例：合法事件、零预算、缺字段、非法视觉、空筛选追问。"""

    def same_trace(response):
        return all(row['trace_id'] == response['trace_id'] for row in response['trace'])

    rows = [
        case('合法事件', app, event('contract-ok', '给朋友推荐礼物，预算300元以内'), 'ok', same_trace),
        case('边界预算零', app,
             event('contract-zero', payload={'slots': {'budget_max': 0}}, session='zero'), 'empty',
             lambda r: r['items'] == [] and r['context']['constraints'] == {'budget_max': 0}),
    ]
    for field in ('trace_id', 'source'):
        invalid = event('contract-missing-' + field)
        del invalid[field]
        rows.append(case('缺少 ' + field, app, invalid, 'error',
                         lambda r: r['error']['code'] == 'INVALID_EVENT'))
    rows.append(case('视觉非法置信度', app,
                     event('contract-bad-label', source='vision',
                           payload={'labels': [{'label': '红色', 'confidence': 1.1}]}),
                     'error', lambda r: r['error']['code'] == 'INVALID_EVENT'))
    rows.append(case('空筛选追问', app,
                     event('contract-filter', payload={'intent': 'filter'}, session='filter'),
                     'needs_clarification',
                     lambda r: r['context']['constraints'] == {} and bool(r['message'])))
    return rows


def _request_roundtrip():
    """跨模块推荐请求必须能无损往返 JSON。"""
    source = event('request-contract')
    request = RecommendationRequest(source['trace_id'], source['user_id'], source['session_id'],
                                    source['timestamp'], source['source'], 'discover',
                                    constraints={'budget_max': 300, 'currency': 'GBP'})
    restored = RecommendationRequest.from_dict(request.to_dict()).to_dict()
    assert restored == request.to_dict(), 'RecommendationRequest JSON 往返不一致'
    return True


def _log_lines(cases, roundtrip):
    """日志写清命令、解释器版本和每条 PASS/FAIL，便于直接贴进报告。"""
    return ['command: python -B scripts/01_contract_test.py',
            'runner: python -B scripts/run_all.py',
            f"python: {sys.version.split()[0]}",
            'report: outputs/stage01_contract_report.json',
            f"request_json_roundtrip: {'PASS' if roundtrip else 'FAIL'}", ''] + \
           [f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'}" for row in cases]


def run(settings):
    """生成阶段1 报告与日志；返回用例列表供 run_all 汇总。"""
    cases = _event_cases(Orchestrator(settings))
    roundtrip = _request_roundtrip()
    write_json(settings, 'stage01_contract_report.json', report(cases))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / 'stage01_contract_test.log').write_text(
        '\n'.join(_log_lines(cases, roundtrip)) + '\n', encoding='utf-8')
    finish(settings, cases, '阶段01')
    return cases


if __name__ == '__main__':
    run(demo_settings())
