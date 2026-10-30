"""总验收入口：依次调用 01~06 阶段脚本的 run()，汇总用例统计并落盘。"""
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _stage import demo_settings, write_json  # noqa: E402

STAGES = ('01_contract_test', '02_context_test', '03_orchestrator_demo',
          '04_explanation_test', '05_feedback_ab', '06_e2e_acceptance')


def run_all(settings=None):
    """逐个阶段执行；单个阶段失败只记录不中断，最后统一落盘并返回汇总。"""
    base = settings or demo_settings()
    stages = {}
    for name in STAGES:
        key = 'stage' + name[:2]
        try:
            cases = importlib.import_module(name).run(base)
            passed = sum(bool(c.get('passed')) for c in cases)
            stages[key] = {'case_count': len(cases), 'passed': passed, 'failed': len(cases) - passed}
        except (SystemExit, Exception) as exc:
            # 阶段脚本内部用非零状态报告失败；记录后继续跑，保证汇总覆盖 6 个阶段。
            message = ('阶段脚本以非零状态退出' if isinstance(exc, SystemExit)
                       else f'{type(exc).__name__}: {exc}')
            stages[key] = {'case_count': 0, 'passed': 0, 'failed': 1, 'fatal': True, 'error': message}
        row = stages[key]
        if row.get('fatal'):    # 正常阶段脚本自己会打印一行，这里只补失败信息
            print(f"{key}: 失败（{row['error']}）")
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(), **stages}
    summary.update({'total_case_count': sum(r['passed'] + r['failed'] for r in stages.values()),
                    'total_passed': sum(r['passed'] for r in stages.values()),
                    'total_failed': sum(r['failed'] for r in stages.values())})
    write_json(base, 'acceptance_summary.json', summary)
    return summary


def main():
    """跑全部阶段并在有失败时非零退出。"""
    summary = run_all()
    print(f"总验收: {summary['total_passed']}/{summary['total_case_count']} PASS")
    if summary['total_failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
