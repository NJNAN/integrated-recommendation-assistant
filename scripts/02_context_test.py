"""阶段2：多模态输入与意图/会话的固定样例验收（≥20 条固定矩阵 + 4 条多轮）。"""
from datetime import datetime, timezone

from _stage import case, demo_settings, event, finish, metrics, write_json
from assistant.orchestrator import Orchestrator


def _check(response, slots, intent, preferences, name):
    """一条固定样例的完整断言：意图 trace + 会话约束/偏好 + 低置信标签/反馈符号。"""
    data = next(item['data'] for item in response['trace'] if item['stage'] == 'intent')
    passed = (response['context'].get('constraints', {}) == slots
              and data['intent'] == intent
              and response['context'].get('preferences', {}) == preferences)
    if name == '弱视觉':
        passed = passed and data['ignored_labels'] == [{'label': '红色', 'confidence': .3}]
    if name in ('点赞', '倒赞'):
        passed = passed and data['feedback'] == (1 if name == '点赞' else -1)
    return passed


def _multi_cases(settings):
    """4 条多轮用例：更新、冲突不污染、撤销、结束清理后恢复基线。"""
    app = Orchestrator(settings)
    app.handle(event('context-seed'))
    app.handle(event('context-multi-1', '背包预算300以内'))
    updated = {'category': '背包', 'budget_max': 300}
    cases = []
    row = case('多轮更新', app, event('context-multi-2', '蓝色'), 'ok',
               lambda r: r['context']['constraints'] == {**updated, 'color': '蓝色'})
    row.update(expected_intent='filter', expected_slots={**updated, 'color': '蓝色'},
               expected_preferences={}, expected_status='ok')
    cases.append(row)
    row = case('多轮冲突不污染', app, event('context-multi-3', '至少500元'), 'conflict',
               lambda r: (r['context']['constraints'] == {**updated, 'color': '蓝色'}
                          and r['context'].get('preferences') == {}))
    row.update(expected_intent='filter', expected_slots={**updated, 'color': '蓝色'},
               expected_preferences={}, expected_status='conflict')
    cases.append(row)
    def undo_ok(response, expected=updated):
        return (response['context']['constraints'] == expected
                and 'color' not in response['context']['constraints'])
    row = case('撤销', app, event('context-undo', payload={'action': 'undo'}), 'ok', undo_ok)
    row.update(expected_intent='discover', expected_slots=updated,
               expected_preferences={}, expected_status='ok')
    cases.append(row)
    ended_event = event('context-end', payload={'action': 'end'})
    ended = app.handle(ended_event)
    after_event = event('context-after-end', '推荐礼物')
    after = app.handle(after_event)
    passed = (ended['status'] == 'ended' and ended['context'] == {} and after['status'] == 'ok'
              and after['context']['constraints'] == {} and after['context']['preferences'] == {}
              and bool(after['items']) and all(i['trace_id'] == after['trace_id'] for i in after['items']))
    cases.append({'name': '结束清理后新事件恢复基线', 'input': ended_event, 'expected_status': 'ended',
                  'expected_intent': 'discover', 'expected_slots': {}, 'expected_preferences': {},
                  'response': ended, 'followup_input': after_event, 'followup_response': after,
                  'passed': bool(passed)})
    return cases


def run(settings):
    """跑固定样例矩阵与多轮会话，落盘 JSONL、指标和错误案例说明。"""
    warm = Orchestrator(settings)
    ids = [item['item_id'] for item in warm.handle(event('context-warm'))['items']]
    # 手工指定的期望矩阵：期望值不来自解析器自身，避免“自己断言自己”。
    fixtures = [
        ('发现', 'text', {'text': '推荐礼物'}, 'discover', {}, 'ok'),
        ('背包别名', 'text', {'text': '推荐双肩包'}, 'filter', {'category': '背包'}, 'ok'),
        ('预算上限', 'text', {'text': '预算300元以内'}, 'filter', {'budget_max': 300}, 'ok'),
        ('预算下限', 'text', {'text': '至少100元'}, 'filter', {'budget_min': 100}, 'ok'),
        ('区间', 'text', {'text': '预算100到300元'}, 'filter', {'budget_min': 100, 'budget_max': 300}, 'ok'),
        ('蓝色', 'text', {'text': '蓝色'}, 'filter', {'color': '蓝色'}, 'ok'),
        ('对象', 'text', {'text': '给朋友买礼物'}, 'filter', {'recipient': '朋友'}, 'ok'),
        ('场景', 'text', {'text': '学习用的台灯'}, 'filter', {'category': '台灯', 'scene': '学习'}, 'ok'),
        ('杯子别名', 'text', {'text': '保温杯'}, 'filter', {'category': '杯子'}, 'ok'),
        ('语音预算', 'voice', {'transcript': '预算200以内的耳机'}, 'filter',
         {'category': '耳机', 'budget_max': 200}, 'ok'),
        ('可信视觉', 'vision', {'labels': [{'label': '背包', 'confidence': .95}]}, 'filter',
         {'category': '背包'}, 'ok'),
        ('视觉融合', 'vision', {'text': '预算300以内',
                                'labels': [{'label': '背包', 'confidence': .9},
                                           {'label': '蓝色', 'confidence': .8}]},
         'filter', {'category': '背包', 'color': '蓝色', 'budget_max': 300}, 'ok'),
        ('弱视觉', 'vision', {'labels': [{'label': '红色', 'confidence': .3}]}, 'discover', {},
         'needs_clarification'),
        ('视觉冲突', 'vision', {'text': '蓝色', 'labels': [{'label': '红色', 'confidence': .99}]},
         'filter', {}, 'conflict'),
        ('点赞', 'gesture', {'gesture': 'thumbs_up', 'item_id': ids[-1]}, 'feedback', {}, 'ok'),
        ('倒赞', 'gesture', {'gesture': 'thumbs_down', 'item_id': ids[0]}, 'feedback', {}, 'ok'),
        ('缺少反馈对象', 'gesture', {'gesture': 'thumbs_up'}, 'feedback', {}, 'needs_clarification'),
        ('相似', 'text', {'intent': 'similar', 'item_id': ids[0]}, 'similar', {}, 'ok'),
        ('比较', 'text', {'intent': 'compare', 'item_ids': ids[:2]}, 'compare', {}, 'ok'),
        ('解释', 'text', {'intent': 'explain', 'item_id': ids[0]}, 'explain', {}, 'ok'),
        ('缺少相似对象', 'text', {'text': '找类似的'}, 'similar', {}, 'needs_clarification'),
        ('缺少比较对象', 'text', {'text': '比较一下'}, 'compare', {}, 'needs_clarification'),
        ('预算冲突', 'text', {'slots': {'budget_min': 500, 'budget_max': 100}}, 'filter', {}, 'conflict'),
        ('低预算空集', 'text', {'text': '耳机预算1元以内'}, 'filter', {'category': '耳机', 'budget_max': 1},
         'empty'),
        ('手势发现', 'gesture', {'gesture': 'victory'}, 'discover', {}, 'ok'),
        ('预算未解析', 'text', {'text': '耳机预算三百元'}, 'filter', {}, 'needs_clarification'),
    ]
    expected_preferences = {'点赞': {ids[-1]: 1}, '倒赞': {ids[0]: -1}}
    cases = []
    for index, (name, source, payload, intent, slots, status) in enumerate(fixtures):
        app = Orchestrator(settings)
        app.handle(event('context-seed'))       # 每条固定样例都从同一基线会话开始
        preferences = expected_preferences.get(name, {})
        row = case(name, app, event(f'context-{index:02d}', source=source, payload=payload), status,
                   lambda r, name=name, slots=slots, intent=intent, prefs=preferences:
                   _check(r, slots, intent, prefs, name))
        row.update(expected_intent=intent, expected_slots=slots,
                   expected_preferences=preferences, expected_status=status)
        cases.append(row)
    cases.extend(_multi_cases(settings))
    write_json(settings, 'stage02_context_cases.jsonl', cases, jsonl=True)
    summary = metrics(cases)
    write_json(settings, 'stage02_context_metrics.json', {
        'generated_at': datetime.now(timezone.utc).isoformat(), 'metrics': summary,
        'minimum_pass_rate': .8, 'failed_cases': [c['name'] for c in cases if not c['passed']]})
    lines = ['# 阶段02 错误案例分析', '',
             '- 重跑: `python -B scripts/02_context_test.py`（或 `python -B scripts/run_all.py`）',
             '- 证据: `outputs/stage02_context_cases.jsonl`、`outputs/stage02_context_metrics.json`',
             f"- 规模: case_count={summary['case_count']}，passed={summary['passed']}，"
             f"failed={summary['failed']}，pass_rate={summary['pass_rate']}", '',
             '## 错误、追问与回滚案例', '',
             '| 分类 | 用例 | 期望状态 | 实际状态 | 消息/错误码 | 判定 |',
             '| --- | --- | --- | --- | --- | --- |']
    categories = {'conflict': '冲突', 'needs_clarification': '追问', 'empty': '空集', 'ended': '回滚/清理'}
    for row in cases:
        if row['expected_status'] == 'ok':
            continue
        response = row['response']
        detail = response['error']['code'] if response.get('error') else response.get('message', '')
        lines.append(f"| {categories.get(row['expected_status'], '错误')} | {row['name']} "
                     f"| {row['expected_status']} | {response['status']} | {detail} "
                     f"| {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ['', '## 失败出口说明', '',
              '- 冲突：同轮预算上下界矛盾或视觉标签与文字颜色矛盾返回 `conflict`，不写入约束；'
              '多轮冲突保持上一轮有效约束，失败请求不污染会话。',
              '- 追问：低置信标签被忽略、缺少反馈/相似/比较对象、预算无法解析时返回 `needs_clarification`，'
              '只报告缺失字段，不猜测用户意图。',
              '- 空集：低预算等硬约束无商品时返回 `empty`，提示“调整条件”，不擅自放宽约束。',
              '- 回滚/清理：冲突后约束不变；结束会话后新事件重建空约束、空偏好的会话（见 JSONL 的 followup_response）。',
              '', '## 复核方法', '', '```powershell', 'python -B scripts/02_context_test.py', '```', '']
    (settings.output_dir / 'stage02_error_cases.md').write_text('\n'.join(lines), encoding='utf-8')
    finish(settings, cases, '阶段02')
    return cases


if __name__ == '__main__':
    run(demo_settings())
