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
    """生成离线演示配置：强制用本地目录，清掉推荐服务凭据。
    例如跑 05/06 时不读本机账号，证据可复现。
    """
    return replace(Settings(), catalog_path=PROJECT_ROOT / 'data/catalog.json', recommender_url='',
                   recommender_username='', recommender_password='', recommender_accounts=())


def event(name, text=None, source='text', payload=None, session='demo', user='demo-user'):
    """构造一条固定事件，trace_id 直接取用例名方便对照。
    例如 event('e2e-voice') 的 trace_id 就是 e2e-voice。
    """
    return {'trace_id': name, 'user_id': user, 'session_id': session,
            'timestamp': '2026-10-08T10:00:00+08:00', 'source': source,
            'payload': payload if payload is not None else {'text': text or '推荐礼物'}}


def write_json(settings, name, content, jsonl=False):
    """把证据写进输出目录，jsonl=True 时按每行一条序列化。
    例如用例明细用 jsonl 便于逐条比对，汇总则用缩进 JSON。
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / name
    if jsonl:
        text = '\n'.join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in content) + '\n'
    else:
        text = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(text, encoding='utf-8')
    return path


def metrics(cases):
    """统计用例数、通过率与 p50/p95/max 延迟。
    例如 7 条用例里 5 条带 elapsed_ms，延迟样本数就是 5。
    """
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
    """把用例列表包装成带生成时间与指标的标准报告。
    例如 05/06 脚本直接拿它当落盘 JSON 的骨架。
    """
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'metrics': metrics(cases), 'cases': cases}


def case(name, app, input_event, expected_status, checks=None):
    """跑一条用例：状态码对得上且自定义断言通过才算 PASS。
    例如期望 error 却返回 ok，passed 就是 False。
    """
    response = app.handle(input_event)
    passed = response['status'] == expected_status and (checks(response) if checks else True)
    return {'name': name, 'input': input_event, 'expected_status': expected_status,
            'response': response, 'passed': bool(passed)}


def finish(settings, cases, label):
    """统一收尾：打印阶段通过数，有失败就以非零状态退出。
    例如打印“阶段06: 7/7 PASS”，否则抛 SystemExit(1)。
    """
    summary = metrics(cases)
    print(f"{label}: {summary['passed']}/{summary['case_count']} PASS; outputs={settings.output_dir}")
    if summary['failed']:
        raise SystemExit(1)
    return summary
