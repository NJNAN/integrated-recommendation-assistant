import sys
sys.path.insert(0, '.')

from dataclasses import replace
from config.settings import Settings, ROOT
from assistant.module_one import integrated_settings
from assistant.adapters import HTTPRecommender
from assistant.contracts import RecommendationRequest

base = Settings()
active, service = integrated_settings(base)
print('接入模式:', '真实模块一' if service else '离线')
if service:
    service.start()
try:
    req = RecommendationRequest('probe', '14911.0', 's', '2026-10-08T10:00:00+08:00', 'text', 'discover')
    rows = HTTPRecommender(active).recall(req)
    scores = [r['score'] for r in rows]
    print('候选数:', len(rows))
    print('score 范围: min=%.6f max=%.6f' % (min(scores), max(scores)))
    print('前 5 条:', [(r['item_id'], round(r['score'], 6)) for r in rows[:5]])
    print('相邻分差（前 10 条）:', [round(scores[i] - scores[i+1], 6) for i in range(9)])
    # 该用户在这份排序里有多少条（决定相邻分差大小）
    import csv
    from pathlib import Path
    grouped = {}
    with (active.module_one_root / 'release/expected/submission_v2_R1_head48.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row['user_id'], 0)
            grouped[row['user_id']] += 1
    print('14911.0 的排序条数:', grouped.get('14911.0'), ' → 相邻归一化分差约 1/(n-1) = %.6f' % (1 / max(1, grouped.get('14911.0') - 1)))
finally:
    if service:
        service.close()
