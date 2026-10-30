"""测试入口：unittest discover 跑 tests，结果与日志落盘，失败非零退出。"""
import io
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stage import write_json  # noqa: E402
from config.settings import ROOT, settings  # noqa: E402


def main():
    """跑全量单元测试并写 outputs/test_results.json 与 test_results.log。"""
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_json(settings, 'test_results.json', {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'successful': result.wasSuccessful(),
        'command': 'python -B scripts/test_report.py',
    })
    (settings.output_dir / 'test_results.log').write_text(stream.getvalue(), encoding='utf-8')
    print(stream.getvalue(), end='')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
