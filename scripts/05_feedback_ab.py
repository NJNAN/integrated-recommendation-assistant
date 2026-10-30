"""阶段05：反馈闭环 A/B 验收——同一用户、查询、预算与 TopK，只让反馈变化。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stage import demo_settings, event, finish, report, write_json  # noqa: E402
from assistant.orchestrator import Orchestrator  # noqa: E402

QUERY = {'text': '推荐300元以内的礼物'}


def _rank(response, item_id):
    """目标商品在结果中的名次；不在结果里返回 None。"""
    ids = [item['item_id'] for item in response['items']]
    return ids.index(item_id) + 1 if item_id in ids else None


def _score(response, item_id):
    """目标商品的分数；不在结果里返回 None。"""
    return next((item['score'] for item in response['items'] if item['item_id'] == item_id), None)


def _scored(response):
    """完整排序结果，用于断言重复反馈不改变任何分数。"""
    return [(item['item_id'], item['score']) for item in response['items']]


def _case(name, input_event, response, expected_status, passed, reason, **extra):
    """组装一条带排名/分数证据的 A/B 用例。"""
    return {'name': name, 'input': input_event, 'expected_status': expected_status,
            'response': response, 'primary_reason': reason, 'passed': bool(passed), **extra}


def _write_markdown(settings, result, ranks, rank_delta, score_delta):
    """写反馈对照表与性能摘要：只留答辩要讲的结论与统计口径。"""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    table = ['# 反馈前后对照', '',
             f'- generated_at: {result["generated_at"]}',
             f'- 目标: `{result["target_item_id"]}`（基线第 {ranks["before"]} 名，取中间排名观察双向变化）',
             f'- 同一候选集合: {result["fixed_candidate_pool"]}',
             f'- 对照范围: {result["comparison_scope"]}', '',
             '| 条件 | 排名 | rank_delta | 目标分数变化 | 原因 |',
             '| --- | --- | --- | --- | --- |',
             f'| 无反馈 | {ranks["before"]} | 0 | 0 | 基线（无偏好） |',
             f'| 正反馈 | {ranks["positive"]} | {rank_delta["positive"]:+d} | {score_delta["positive"]:+.6g} |'
             ' feedback 权重 +2.5；同标签商品 tag_affinity 正影响 |',
             f'| 负反馈 | {ranks["negative"]} | {rank_delta["negative"]:+d} | {score_delta["negative"]:+.6g} |'
             ' feedback 权重 -2.5 覆盖正反馈 |',
             '', '重复同方向反馈不累计权重；结束或切换会话恢复基线。',
             '证据：`outputs/stage05_feedback_ab.json`（rank_change/rank_delta/score_delta）。']
    (settings.output_dir / 'stage05_feedback_table.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    metrics = result['metrics']
    summary = ['# 阶段05 性能与失败摘要', '',
               f'- generated_at: {result["generated_at"]}',
               '- 重跑: `python -B scripts/05_feedback_ab.py`（或 `python -B scripts/run_all.py`）',
               '- 证据: `stage05_feedback_ab.json`、`stage05_feedback_table.md`', '',
               '## 性能', '', '| 指标 | 值 |', '| --- | --- |',
               f'| 用例数 | {metrics["case_count"]} |',
               f'| 通过 / 失败 | {metrics["passed"]} / {metrics["failed"]} |',
               f'| p50 / p95 / max 延迟(ms) | {metrics["latency_ms"]["p50"]} /'
               f' {metrics["latency_ms"]["p95"]} / {metrics["latency_ms"]["max"]} |',
               f'| 计时样本数 | {metrics["latency_ms"]["sample_count"]} |',
               '',
               '- 计时口径：单次响应 elapsed_ms，含会话合并、召回排序与逐条解释。', '',
               '## 失败与降级', '', '| 类型 | 数量/说明 | 证据 |', '| --- | --- | --- |',
               '| 预期错误 | 1：非法反馈 `INVALID_FEEDBACK`，按预期通过 | `expected_error_count=1` |',
               f'| 意外失败 | {result["unexpected_error_count"]} | `unexpected_error_count` |']
    (settings.output_dir / 'stage05_performance_summary.md').write_text(
        '\n'.join(summary) + '\n', encoding='utf-8')


def run(settings):
    """跑 5 条 A/B 用例并落盘 JSON 与两个 md。"""
    app = Orchestrator(settings)
    payload = {**QUERY, 'top_k': settings.max_top_k}
    ev_before = event('feedback-before', payload=payload)
    before = app.handle(ev_before)
    if not before['items']:
        raise SystemExit('阶段05 失败：基线查询没有返回商品，无法做 A/B 对照')
    middle = len(before['items']) // 2
    target = before['items'][middle]['item_id']
    ev_up = event('feedback-up', source='gesture', payload={'gesture': 'thumbs_up', 'item_id': target})
    ev_repeat = event('feedback-repeat', source='gesture',
                      payload={'gesture': 'thumbs_up', 'item_id': target})
    ev_down = event('feedback-down', source='gesture', payload={'gesture': 'thumbs_down', 'item_id': target})
    ev_invalid = event('feedback-invalid', source='gesture',
                       payload={'gesture': 'thumbs_down', 'item_id': 'nonexistent-item'})
    ev_isolated = event('feedback-isolation', payload=payload, session='fresh')
    up, repeat = app.handle(ev_up), app.handle(ev_repeat)
    down, invalid, isolated = app.handle(ev_down), app.handle(ev_invalid), app.handle(ev_isolated)

    ranks = {'before': _rank(before, target), 'positive': _rank(up, target), 'negative': _rank(down, target)}
    scores = {'before': _score(before, target), 'positive': _score(up, target),
              'negative': _score(down, target)}
    same_pool = (_scored(before) and sorted(i['item_id'] for i in before['items'])
                 == sorted(i['item_id'] for i in up['items'])
                 == sorted(i['item_id'] for i in down['items']))
    rank_delta = {'positive': ranks['positive'] - ranks['before'],
                  'negative': ranks['negative'] - ranks['before']}
    score_delta = {'positive': round(scores['positive'] - scores['before'], 6),
                   'negative': round(scores['negative'] - scores['before'], 6)}
    rollback = (invalid['context']['preferences'] == down['context']['preferences']
                and invalid['context']['constraints'] == down['context']['constraints']
                and invalid['context']['last_item_ids'] == down['context']['last_item_ids'])
    cases = [
        _case('正反馈提升', ev_up, up, 'ok',
              ranks['positive'] < ranks['before'] and same_pool and scores['positive'] > scores['before'],
              '目标商品 feedback +2.5；同标签商品 tag_affinity 正影响',
              rank=ranks['positive'], rank_delta=rank_delta['positive'], score_delta=score_delta['positive']),
        _case('重复反馈无变化', ev_repeat, repeat, 'ok', _scored(up) == _scored(repeat),
              '同方向反馈覆盖旧值，权重不累积',
              rank=ranks['positive'], rank_delta=0, score_delta=0),
        _case('反向反馈降低', ev_down, down, 'ok',
              ranks['negative'] > ranks['before'] and ranks['negative'] > ranks['positive']
              and scores['negative'] < scores['before'] and down['context']['preferences'].get(target) == -1,
              '目标商品 feedback -2.5 覆盖正反馈；同标签商品 tag_affinity 转负',
              rank=ranks['negative'], rank_delta=rank_delta['negative'], score_delta=score_delta['negative']),
        _case('非法反馈不污染', ev_invalid, invalid, 'error',
              invalid['error']['code'] == 'INVALID_FEEDBACK' and rollback,
              '目标不在当前会话最近展示集：返回 INVALID_FEEDBACK 且会话回滚到上一次',
              rank=None, rank_delta=None, score_delta=None),
        _case('会话隔离', ev_isolated, isolated, 'ok',
              _scored(isolated) == _scored(before) and not isolated['context']['preferences'],
              '新会话无偏好，排序恢复无反馈基线',
              rank=_rank(isolated, target),
              rank_delta=_rank(isolated, target) - ranks['before'], score_delta=0),
    ]
    result = report(cases)
    result.update({'target_item_id': target,
                   'target_selection': {'policy': 'baseline_middle_rank', 'item_count': len(before['items']),
                                        'index': middle, 'baseline_rank': ranks['before']},
                   'fixed_candidate_pool': bool(same_pool), 'rank_change': ranks,
                   'rank_delta': rank_delta, 'score_delta': score_delta,
                   'negative_below_baseline': ranks['negative'] > ranks['before'],
                   'comparison_scope': '同一用户、同一查询（推荐300元以内的礼物，预算上限300元）、同一目录、'
                                       'TopK=24；仅反馈变化',
                   'expected_error_count': 1,
                   'unexpected_error_count': sum(not c['passed'] for c in cases)})
    write_json(settings, 'stage05_feedback_ab.json', result)
    _write_markdown(settings, result, ranks, rank_delta, score_delta)
    finish(settings, cases, '阶段05')
    return cases


if __name__ == '__main__':
    run(demo_settings())
