"""六阶段脚本共用的小工具：路径引导、固定事件、落盘与指标。

脚本可以 `python -B scripts/0X_xxx.py` 单独运行，也可以被 run_all.py 导入后调用 run()。
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from config.settings import Settings, ROOT as PROJECT_ROOT  # noqa: E402  (需要先补好 sys.path)


def demo_settings():
    """阶段证据固定使用离线合成目录，不受本机凭据和外部服务影响。"""
    return replace(Settings(), catalog_path=PROJECT_ROOT / 'data/catalog.json', recommender_url='',
                   recommender_username='', recommender_password='', recommender_accounts=())


def event(name, text=None, source='text', payload=None, session='demo', user='demo-user'):
    """构造一条固定事件：trace_id 直接用用例名，便于在证据里对照。"""
    return {'trace_id': name, 'user_id': user, 'session_id': session,
            'timestamp': '2026-10-08T10:00:00+08:00', 'source': source,
            'payload': payload if payload is not None else {'text': text or '推荐礼物'}}


def write_json(settings, name, content, jsonl=False):
    """落盘证据；jsonl=True 时每行一条，便于逐条对照。"""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / name
    if jsonl:
        text = '\n'.join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in content) + '\n'
    else:
        text = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(text, encoding='utf-8')
    return path


def metrics(cases):
    """通过率与延迟分位数；只统计真正有响应计时的用例。"""
    timings = sorted(c['response']['elapsed_ms'] for c in cases if 'elapsed_ms' in c.get('response', {}))
    count = len(cases)
    passed = sum(bool(c['passed']) for c in cases)
    sample = len(timings)
    return {'case_count': count, 'passed': passed, 'failed': count - passed,
            'pass_rate': round(passed / count, 4) if count else 0,
            'latency_ms': {'sample_count': sample,
                           'p50': timings[max(0, math.ceil(sample * .5) - 1)] if sample else None,
                           'p95': timings[max(0, math.ceil(sample * .95) - 1)] if sample else None,
                           'max': max(timings, default=None)}}


def report(cases):
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'metrics': metrics(cases), 'cases': cases}


def case(name, app, input_event, expected_status, checks=None):
    """跑一条用例：状态一致 + 自定义断言都通过才算 PASS。"""
    response = app.handle(input_event)
    passed = response['status'] == expected_status and (checks(response) if checks else True)
    return {'name': name, 'input': input_event, 'expected_status': expected_status,
            'response': response, 'passed': bool(passed)}


def finish(settings, cases, label):
    """统一收尾：打印结果，有失败就以非零状态退出（任务书要求可重跑）。"""
    summary = metrics(cases)
    print(f"{label}: {summary['passed']}/{summary['case_count']} PASS; outputs={settings.output_dir}")
    if summary['failed']:
        raise SystemExit(1)
    return summary
