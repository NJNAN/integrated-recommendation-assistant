"""阶段4：解释的证据约束——正常事实，以及无证据/字段缺失/价格不一致/弱视觉标签四种异常。"""
from copy import deepcopy
import time

from _stage import case, demo_settings, event, finish, report, write_json
from assistant.contracts import RecommendationItem
from assistant.explanation import explain
from assistant.orchestrator import Orchestrator


def _write_text(settings, name, lines):
    """对照表与异常记录是 Markdown；_stage 只负责 JSON，这里补一个纯文本写出。"""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _citations_ok(item):
    """引用必须可回溯：都在本商品 snippets 内，且 id 以 catalog:<item_id>: 开头。"""
    ids = {snippet['id'] for snippet in item['evidence']['snippets']}
    prefix = f"catalog:{item['item_id']}:"
    citations = item['explanation']['citations']
    return bool(citations) and set(citations) <= ids and all(c.startswith(prefix) for c in citations)


def _normal_ok(response):
    """正常解释：价格写进正文、事实与证据一致、引用可核验、语音文本相同。"""
    items = response['items']
    return (response['status'] == 'ok' and bool(items)
            and all(i['trace_id'] == response['trace_id'] for i in items)
            and all(f"¥{i['price']:g}" in i['explanation']['text'] for i in items)
            and all(i['price'] == i['evidence']['attributes']['price'] for i in items)
            and all(i['explanation']['speech_text'] == i['explanation']['text'] for i in items)
            and all(_citations_ok(i) for i in items))


def _drop_snippets(target):
    target.evidence.snippets = []
    return {'field': 'evidence.snippets', 'value': [], 'expected_code': 'INSUFFICIENT_EVIDENCE'}


def _drop_price(target):
    del target.evidence.attributes['price']
    return {'field': 'evidence.attributes.price', 'value': '<deleted>', 'expected_code': 'EVIDENCE_MISMATCH'}


def _shift_price(target):
    target.evidence.attributes['price'] += 1
    return {'field': 'evidence.attributes.price', 'value': '<original+1>', 'expected_code': 'EVIDENCE_MISMATCH'}


def _inject(name, item, inject):
    """在深拷贝上注入异常，断言解释降级、引用清空且没有编造内容。"""
    target = deepcopy(item)
    expected = inject(target)
    started = time.perf_counter()
    result = explain(target)
    return {'name': name, 'input': {'trace_id': item.trace_id, 'item_id': item.item_id},
            'injection': expected, 'expected_status': 'degraded', 'expected_code': expected['expected_code'],
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 3), 'response': result,
            'passed': bool(result['code'] == expected['expected_code'] and result['degraded']
                           and not result['citations'])}


def run(settings):
    """跑 5 条用例并落盘三类证据；返回用例列表供 run_all 汇总。"""
    normal_event = event('explanation-normal', '推荐台灯')
    cases = [case('正常事实解释', Orchestrator(settings), normal_event, 'ok', _normal_ok)]
    items = cases[0]['response']['items']
    if not items:
        raise SystemExit('阶段4 需要至少一条推荐结果，请先确认阶段3 推荐链路可用')
    item = RecommendationItem.from_dict(next((i for i in items if i['price'] is not None), items[0]))
    cases.append(_inject('无证据不编造', item, _drop_snippets))
    cases.append(_inject('商品字段缺失', item, _drop_price))
    cases.append(_inject('价格不一致', item, _shift_price))
    weak_ok = lambda r: bool(r['items']) and all(not i['evidence']['visual_labels'] for i in r['items'])
    cases.append(case('弱视觉商品标签忽略', Orchestrator(settings), event('explanation-low-vision', '绿色台灯'),
                      'ok', weak_ok))
    write_json(settings, 'stage04_explanation_cases.jsonl', cases, True)
    generated_at = report(cases)['generated_at']
    normal_items = cases[0]['response']['items']
    table = ['# 解释与证据对照表', '', f'- generated_at: {generated_at}',
             '- 证据路径: `outputs/stage04_explanation_cases.jsonl`',
             '', '| 商品 | 价格 | 解释 | 引用 |', '| --- | --- | --- | --- |']
    table += [f"| {i['name']} | {i['price']} | {i['explanation']['text']} "
              f"| {', '.join(i['explanation']['citations'])} |" for i in normal_items]
    _write_text(settings, 'stage04_evidence_table.md', table)
    abnormal = ['# 阶段04 异常解释记录', '', f'- generated_at: {generated_at}',
                '- 重跑命令: `python -B scripts/04_explanation_test.py`',
                '- 证据路径: `outputs/stage04_explanation_cases.jsonl`、`outputs/stage04_evidence_table.md`', '',
                '| 用例 | 注入 | 期望错误码 | 实际错误码 | 引用数 | 判定 |',
                '| --- | --- | --- | --- | --- | --- |']
    for row in cases[1:]:
        injection = row.get('injection')
        detail = f"{injection['field']}={injection['value']}" if injection else '低置信视觉标签（0.7 阈值以下忽略）'
        result = row['response']
        abnormal.append(f"| {row['name']} | {detail} | {row.get('expected_code', 'OK')} "
                        f"| {result.get('code', row['expected_status'])} | {len(result.get('citations', []))} "
                        f"| {'PASS' if row['passed'] else 'FAIL'} |")
    abnormal += ['', '## 降级原则', '',
                 '- 缺证据片段或属性时停止解释并返回 `INSUFFICIENT_EVIDENCE`，不编造引用。',
                 '- 属性与证据不一致（缺字段、价格整加一）时返回 `EVIDENCE_MISMATCH`，引用清空。',
                 '- 视觉标签低于阈值不进入证据，正常解释继续；异常用例均记录注入内容与判定。']
    _write_text(settings, 'stage04_abnormal_explanations.md', abnormal)
    finish(settings, cases, '阶段04')
    return cases


if __name__ == '__main__':
    run(demo_settings())
