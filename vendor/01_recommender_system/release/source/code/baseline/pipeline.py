"""
课程基线（baseline）端到端运行入口：prepare → train → evaluate，或 all 一次执行。

本文件是六阶段课程交付的“总调度”：它不实现算法，而是把
data.clean_data / data.split_data / data.sequences（阶段01-02）、
engine.train_item2vec / engine.BaselineEngine（阶段03-04）、
LR / LightGBM 精排（阶段05）、Top-K 评估与报告生成（阶段06）串成一条
可一键复现的流水线，并把每个阶段的产物（CSV/JSON/图片/报告）落盘到
outputs/baseline/ 下，供答辩与审计使用。

执行方式（在 code/ 目录下）：
    python -m baseline.pipeline all                 # 一次跑完三个阶段
    python -m baseline.pipeline prepare             # 只做清洗与数据切分
    python -m baseline.pipeline train               # 只做训练（需先 prepare）
    python -m baseline.pipeline evaluate            # 只做评估（需先 train）
    python -m baseline.pipeline all --eval-users 0  # 用全部合格用户评估（默认500抽样）

可靠性约定：
- 三个阶段的“已完成”状态写入 outputs/baseline/run_status.json，
  任何阶段中途失败都会把 complete 置 False，Web 服务据此拒绝加载半成品模型；
- 配置（时间切分点、配额、种子）保存在 data/processed/baseline/config.json，
  evaluate 时校验“加载的模型配置 == 当前配置”，防止旧模型配新数据产生虚假数字。
"""
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import sqlite3
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import Config, MODELS, OUTPUTS, PROCESSED, RAW, ROUTES
from .data import clean_data, sequences, split_data, write_json
from .engine import BaselineEngine, FEATURES, top_indices, train_item2vec
from .metrics import overlap, topk_metrics


def read_frame(name):
    """读取 data/processed/baseline/ 下的一个处理产物。

    - user_id/item_id/invoice_no 一律按字符串读：商品编号含字母与前导零，
      若被 pandas 推断成数字会悄悄丢失（如 '007' → 7），导致后续查不到；
    - transaction_time 列解析为时间类型，方便窗口切分与时间比较；
    - catalog.csv 没有时间列，需单独关闭 parse_dates；
    - 商品目录缺失名称/类目时填空值，保证 join 后列对齐、前端展示不崩溃。

    返回带标准 dtype 的 DataFrame。
    """
    return pd.read_csv(PROCESSED / name, dtype={'user_id': str, 'item_id': str, 'invoice_no': str},
                       parse_dates=['transaction_time'] if name != 'catalog.csv' else None).fillna(
                           {'name': '未知商品', 'description': '', 'category': '未知'})


def read_config():
    """从 data/processed/baseline/config.json 读回 prepare 阶段使用的配置。

    为什么要“落盘再读回”而不是直接用内存里的 Config：
    三个阶段是三个独立进程（命令行分别调用），train/evaluate 启动时并不知道
    prepare 时用了什么切分点/配额/种子；只有把配置序列化到磁盘，
    才能保证三阶段使用完全相同的实验配置，杜绝“评估时悄悄改了配置”的复现风险。
    """
    return Config(**json.loads((PROCESSED / 'config.json').read_text(encoding='utf-8')))


def prepare(cfg):
    """阶段 01-02：清洗原始 CSV → 时间切分 → 订单序列，全部产物落盘。

    流程：
    1. 建目录、写 run_status（preparing, complete=False）——防止中途失败后
       把旧模型误当成“新配置训练结果”；
    2. 读取 data/raw 下原始 transactions/products（latin1 / utf-8 编码，
       保持源文件只读），先写 raw_summary.json（原始行数、缺失用户数、
       国家 Top10、商品 Top10）作为“清洗前快照”；
    3. clean_data() 做三级清洗漏斗，把清洗结果、商品目录、异常明细、
       逐步保留率（stage01_retention.csv）分别落盘；商品目录同时写入
       SQLite（products.sqlite3）供 Web 服务按名称/类目查询；
    4. split_data() 按 history_end/test_start 切出 history / rank_train / test
       三个窗口，各窗口的行数/用户数/起止时间记入 data_summary.json；
    5. sequences() 把 history 期交易聚合成“一订单一句”的序列表
       （user_sequences.csv），并输出长度统计与抽样示例；
    6. 写 manifest.json：配置、Python 版本、关键依赖版本、原始文件 SHA-256，
       作为“这份产物是用什么环境生成的”审计凭据。

    本阶段不做任何训练，但它是后续 train 阶段所有特征的唯一数据来源。
    """
    for p in [PROCESSED, OUTPUTS, MODELS]:
        p.mkdir(parents=True, exist_ok=True)
    # 防止中途失败后误把旧模型当作新配置训练结果。
    write_json(OUTPUTS / 'run_status.json', {'stage': 'preparing', 'complete': False})
    tx = pd.read_csv(RAW / 'transactions.csv', encoding='latin1',
                     dtype={'InvoiceNo': 'string', 'StockCode': 'string', 'CustomerID': 'string'})
    products = pd.read_csv(RAW / 'products.csv', encoding='utf-8', dtype={'StockCode': 'string'})
    write_json(OUTPUTS / 'raw_summary.json', {
        'rows': len(tx), 'users': int(tx.CustomerID.nunique()), 'items': int(tx.StockCode.nunique()),
        'missing_customer_id': int(tx.CustomerID.isna().sum()),
        'country_top10': tx.Country.value_counts().head(10).to_dict(),
        'item_top10': tx.StockCode.value_counts().head(10).to_dict()})
    clean, catalog, retention, invalid = clean_data(tx, products)
    clean.to_csv(PROCESSED / 'clean_transactions.csv', index=False)
    catalog.to_csv(PROCESSED / 'catalog.csv', index=False)
    with sqlite3.connect(PROCESSED / 'products.sqlite3') as con:
        catalog.to_sql('products', con, if_exists='replace', index=False)
        con.execute('CREATE UNIQUE INDEX IF NOT EXISTS product_id ON products(item_id)')
    invalid.to_csv(PROCESSED / 'invalid_transactions.csv', index=False)
    retention.to_csv(OUTPUTS / 'stage01_retention.csv', index=False)
    clean.groupby('Country').size().sort_values(ascending=False).head(10).to_csv(OUTPUTS / 'country_top10.csv')
    clean.groupby('item_id').size().sort_values(ascending=False).head(10).to_csv(OUTPUTS / 'item_top10.csv')
    # 商品抽查：随机抽最多10个去重商品，供人工核对“清洗后目录是否合理”。
    clean[['item_id', 'name', 'category', 'catalog_match']].drop_duplicates('item_id').sample(
        min(10, clean.item_id.nunique()), random_state=cfg.seed).to_csv(OUTPUTS / 'catalog_spotcheck.csv', index=False)
    parts = split_data(clean, cfg)
    summary = {'raw_rows': len(tx), 'clean_rows': len(clean), 'invalid_rows': len(invalid),
               'unmatched_rows': int((~clean.catalog_match).sum()), 'windows': {}}
    for name, frame in zip(['history', 'rank_train', 'test'], parts):
        frame.to_csv(PROCESSED / f'{name}.csv', index=False)
        summary['windows'][name] = {'rows': len(frame), 'users': int(frame.user_id.nunique()),
                                    'first': str(frame.transaction_time.min()), 'last': str(frame.transaction_time.max())}
    # 订单序列：history 期每张订单内商品去重后作为一个“句子”（Item2Vec 语料）。
    seqs = sequences(parts[0])
    seq_table = pd.DataFrame(seqs)
    # items 是列表，CSV 无法直接存，序列化成一列 JSON 字符串。
    seq_table['items'] = seq_table['items'].map(json.dumps)
    seq_table.to_csv(PROCESSED / 'user_sequences.csv', index=False)
    lengths = np.array([s['length'] for s in seqs])
    write_json(OUTPUTS / 'sequence_quality.json', {'count': len(seqs), 'mean': float(lengths.mean()),
                'singleton_ratio': float((lengths == 1).mean()),
                'quantiles': {str(q): float(np.quantile(lengths, q)) for q in [.25, .5, .75, .95]},
                'examples': [seqs[i] for i in np.random.default_rng(cfg.seed).choice(len(seqs), min(3, len(seqs)), replace=False)]})
    # 配置落盘：train/evaluate 从磁盘读回，保证三个阶段配置一致。
    write_json(PROCESSED / 'config.json', cfg.asdict())
    write_json(OUTPUTS / 'data_summary.json', summary)
    # 版本清单 + 原始文件哈希：产物可审计（换了数据/依赖会导致 manifest 变化）。
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'gensim', 'scikit-learn', 'flask']}
    write_json(OUTPUTS / 'manifest.json', {'config': cfg.asdict(), 'python': platform.python_version(),
        'versions': versions, 'raw_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.glob('*.csv')}})
    print('prepare:', summary, flush=True)


def truths(frame, engine):
    """计算“未来窗口内、历史未购买过的新商品”集合（课程评估口径）。

    参数：
    - frame：rank_train 或 test 的购买记录；
    - engine：已基于 history 构建的引擎，engine.histories 存每个用户的历史商品。

    返回值：{user_id: {未来购买且历史未购的商品集合}}。

    设计理由（为什么 truth 要排除历史已购）：
    课程口径把“推荐”定义为“帮用户发现没买过的新东西”，所以：
    - 用户未来买了 A，但 A 在他历史里已存在 → 这属于“复购/补货”，不算本口径的命中；
    - 分母（Recall）仍保留历史从未出现过的未来商品——它们未被任何模型召回，
      不强行塞进候选，才能如实反映“召回失败”的比例。
    """
    # 评测目标是“历史建模期未购买过的新商品”，分母保留未来冷商品。
    return {str(u): set(g.item_id) - set(engine.histories.get(str(u), [])) for u, g in frame.groupby('user_id')}


def train():
    """阶段 03-05：训练 Item2Vec → 构建引擎（四路召回）→ 训练精排 LR/LightGBM。

    关键步骤：
    1. 读 prepare 阶段产出的 history/catalog/序列；训练 Skip-gram Item2Vec 并保存；
    2. BaselineEngine 基于 history 计算热门榜、ItemCF 近邻、用户向量，保存 engine.pkl；
    3. 对每个有“新购标签”的用户：四路召回融合出候选集 → 与标签窗口真实购买
       求交得正例 → 从候选里未购买的抽负例（正负 1:5）→ 生成特征行；
       注意正例必须“自然被召回”，绝不把未来答案强塞进候选（否则是数据泄漏）；
    4. LR = StandardScaler + LogisticRegression（整体保存，防止推理时漏标准化）；
       LightGBM 可选：装了就训练并记录特征重要性，没装则如实记录“未完成原因”；
    5. 保存 rankers.pkl，并做“保存→重载→相似查询”自检，验证持久化链路；
    6. 写 training.json（样本数/正负例数/训练集 AUC）、demo_ids（演示用户与商品）。
    """
    cfg = read_config()
    write_json(OUTPUTS / 'run_status.json', {'stage': 'training', 'complete': False})
    history, catalog = read_frame('history.csv'), read_frame('catalog.csv')
    seqs = pd.read_csv(PROCESSED / 'user_sequences.csv', dtype={'user_id': str, 'invoice_no': str})
    # 读回时把“JSON 字符串的 items 列”还原成真正的列表。
    records = [{**r, 'items': json.loads(r['items'])} for r in seqs.to_dict('records')]
    started = time.perf_counter()
    # Skip-gram 商品向量：同订单共现的商品在向量空间被拉近。
    w2v = train_item2vec(records, cfg)
    w2v.save(str(MODELS / 'item2vec.model'))
    print(f'Item2Vec: {len(w2v.wv)} items, {time.perf_counter()-started:.1f}s', flush=True)
    # 引擎封装热门/ItemCF/用户向量/融合/特征，之后 evaluate 与 Web 共用。
    engine = BaselineEngine(history, catalog, cfg, w2v)
    engine.save(MODELS)
    # 精排标签 = 标签窗口内购买过、且历史期未购买过的商品（与测试口径一致）。
    label_sets = truths(read_frame('rank_train.csv'), engine)
    rng = np.random.default_rng(cfg.seed)
    samples, recall_rows = [], []
    for n, uid in enumerate(sorted(label_sets)):
        # candidates() 返回（去重融合后的候选、各路原始分、各路排名）。
        candidates, scores, _ = engine.candidates(uid)
        actual = label_sets[uid]
        # 逐用户记录候选清单与来源：用于召回覆盖率诊断（阶段四要求）。
        recall_rows.append({'user_id': uid, 'items': json.dumps([c['item_id'] for c in candidates]),
                            'sources': json.dumps([c['sources'] for c in candidates]), 'length': len(candidates)})
        labels = np.array([int(c['item_id'] in actual) for c in candidates])
        positives, negatives = np.flatnonzero(labels), np.flatnonzero(labels == 0)
        # 正样本必须自然被召回，不把未来答案强塞进候选。
        if len(positives) == 0 or len(negatives) == 0:
            continue  # 无正例或无可抽负例的用户不进监督训练，但仍计入覆盖率
        # 负例抽样：min(全部负例, 正例数 × negative_ratio)，默认 1:5，保证类别均衡可控。
        take = np.sort(np.concatenate([positives, rng.choice(negatives, min(len(negatives), len(positives) * cfg.negative_ratio), replace=False)]))
        x = engine.features(uid, candidates, scores).iloc[take].copy()
        x.insert(0, 'item_id', [candidates[i]['item_id'] for i in take])
        x.insert(0, 'user_id', uid)
        x['label'] = labels[take]
        samples.append(x)
        if n % 500 == 0:
            print('rank samples users:', n, flush=True)
    if not samples:
        raise ValueError('召回集中没有可用正负训练样本')
    pd.DataFrame(recall_rows).to_csv(PROCESSED / 'recall_set.csv', index=False)
    sample = pd.concat(samples, ignore_index=True)
    sample.to_csv(PROCESSED / 'rank_train_samples.csv', index=False)
    # LR 精排：标准化 + 逻辑回归打包成一条 pipeline——fit 时一起学 scaler 参数，
    # predict 时自动先标准化，避免“训练标准化了、推理忘了标准化”的经典 bug。
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.seed))
    lr.fit(sample[FEATURES], sample.label)
    models = {'lr': lr}
    lgb_status = '已训练；与 LR 使用相同样本、特征、候选和测试用户。'
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        # 可选依赖缺失不算失败，但必须记录原因，保证报告诚实。
        lgb_status = f'未完成可选 LightGBM：依赖不可用（{exc}）。必做 LR 已训练。'
    else:
        # 与 LR 完全相同的样本/特征/标签；树模型对量纲不敏感，故不再包 StandardScaler。
        lgb = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=.05,
                            min_child_samples=40, reg_lambda=1.0, random_state=cfg.seed,
                            n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
        lgb.fit(sample[FEATURES], sample.label)
        models['lightgbm'] = lgb
        # 特征重要性（分裂次数）：只说明“哪些特征常被用来切分”，不代表正负方向。
        pd.DataFrame({'feature': FEATURES, 'importance': lgb.feature_importances_}).sort_values(
            'importance', ascending=False).to_csv(OUTPUTS / 'lightgbm_importance.csv', index=False)
    with (MODELS / 'rankers.pkl').open('wb') as f:
        pickle.dump(models, f)  # 字典 {模型名: 模型对象} 整体序列化
    # LR 系数：特征已标准化，系数绝对值可跨特征比较“谁对分数影响大”。
    pd.DataFrame({'feature': FEATURES, 'standardized_coefficient': lr[-1].coef_[0]}).to_csv(OUTPUTS / 'lr_coefficients.csv', index=False)
    write_json(OUTPUTS / 'training.json', {'sample_rows': len(sample), 'sample_users': int(sample.user_id.nunique()),
        'positives': int(sample.label.sum()), 'negatives': int((sample.label == 0).sum()),
        # 训练集 AUC 只用于自检收敛，不代表测试表现（模型见过这些样本）。
        'train_auc': {name: float(roc_auc_score(sample.label, model.predict_proba(sample[FEATURES])[:, 1])) for name, model in models.items()},
        'lightgbm': lgb_status, 'features': FEATURES, 'vocab_size': len(w2v.wv), 'seconds': time.perf_counter() - started})
    # 模型加载后立即核对，验证持久化链路。
    restored = BaselineEngine.load(MODELS)
    # 相似案例自检：取热门商品做 query，看重载引擎能否返回一致结果。
    example_items = [str(engine.ids[i]) for i in top_indices(engine.popularity, engine.ids, len(engine.ids)) if engine.in_vocab[i]][:5]
    examples = [{'query': engine.enrich(i, 1), 'similar': restored.similar(i, 5)} for i in example_items]
    write_json(OUTPUTS / 'similar_cases.json', examples)
    # 演示 ID：订单数最多的两个用户 + 上面的热门商品，供 Web 演示直接使用。
    demo_users = sorted(engine.users, key=lambda u: (-engine.orders[u], u))[:2]
    write_json(OUTPUTS / 'demo_ids.json', {'users': demo_users, 'items': example_items[:2]})
    (OUTPUTS / 'demo_ids.txt').write_text('用户: ' + ', '.join(demo_users) + '\n商品: ' + ', '.join(example_items[:2]) + '\n', encoding='utf-8')
    print('training complete:', len(sample), 'samples', flush=True)


def evaluate():
    """阶段 06：在测试期上对热门/ItemCF/融合+LR/融合+LightGBM 做公平 Top-K 评估。

    公平性设计（课程验收核心）：
    - 用 prepare 保存的 cfg 与 train 保存的模型核对，不一致立即报错；
    - 从“有非空 new-truth 的用户”里按 cfg.seed 抽 cfg.eval_users 人（默认500），
      名单写 test_users.json，保证所有策略、所有对比都用同一批人；
    - 同一用户同一 K：popular / itemcf / LR / LightGBM 的候选集完全一致，
      popular 与 itemcf 各自独立出 Top-K，不允许冷用户 CF 为空时偷偷混入热门；
    - AUC 在所有测试候选上计算（候选条件排序能力）；Top-K 指标包含“召回失败”
      （候选里根本没有真实商品时 Precision=0、Recall=0 如实计）；
    - 召回诊断：比较默认配额 vs 热门配额×2 的候选覆盖率与热门占比；
    - 每个用户记录 truth/候选/各策略排名到 ranking_results.json（失败案例可溯源）；
    - 报告指标 CSV、模型对比图、自动生成的 baseline_report.md 全部落盘，
      全部写完才把 run_status 置 complete=True（Web 依赖该开关）。
    """
    cfg = read_config()
    engine = BaselineEngine.load(MODELS)
    # 配置一致性校验：模型是旧配置训练的就不能拿新配置的口径去评估。
    if engine.cfg != cfg:
        raise ValueError('数据配置与模型不一致，请重新 train')
    with (MODELS / 'rankers.pkl').open('rb') as f:
        models = pickle.load(f)
    write_json(OUTPUTS / 'run_status.json', {'stage': 'evaluating', 'complete': False})
    all_truth = truths(read_frame('test.csv'), engine)
    # eligible = 测试期确有“新购买”的用户；无真实答案的用户无法算命中率。
    eligible = sorted(u for u, truth in all_truth.items() if truth)
    rng = np.random.default_rng(cfg.seed)
    users = sorted(rng.choice(eligible, min(cfg.eval_users or len(eligible), len(eligible)), replace=False).tolist())
    if not users:
        raise ValueError('没有有效测试用户')
    write_json(OUTPUTS / 'test_users.json', users)
    rows, recalls, samples, examples = [], [], [], []
    rank_records = []
    auc_y, auc_preds = [], {name: [] for name in models}
    # 全局热门 Top-K：用于衡量“这个用户的推荐是否只是全局热门的翻版”。
    global_pop = engine.ids[top_indices(engine.popularity, engine.ids, cfg.k)].tolist()
    for n, uid in enumerate(users):
        actual = all_truth[uid]
        candidates, scores, routes = engine.candidates(uid)
        ids = [c['item_id'] for c in candidates]
        x = engine.features(uid, candidates, scores)
        labels = np.array([int(i in actual) for i in ids])
        scored = x.copy()
        scored.insert(0, 'item_id', ids); scored.insert(0, 'user_id', uid); scored['label'] = labels
        samples.append(scored)
        auc_y.extend(labels.tolist())
        # 热门、纯 ItemCF 同一用户、同一 K，均排除历史已购。CF 无结果不偷偷混入热门。
        rankings = {'popular': routes['popular'][:cfg.k], 'itemcf': routes['itemcf'][:cfg.k]}
        for name, model in models.items():
            # predict_proba 输出“候选是正例”的概率，这里只把它当排序分数。
            p = model.predict_proba(x)[:, 1] if len(x) else np.array([])
            auc_preds[name].extend(p.tolist())
            ranks = top_indices(p, np.array(ids), cfg.k)
            rankings[name] = [ids[i] for i in ranks]
        for name, recs in rankings.items():
            rows.append({'user_id': uid, 'strategy': name, **topk_metrics(recs, actual, cfg.k),
                         'popular_overlap': overlap(recs, rankings['popular'], cfg.k),
                         'global_popular_overlap': overlap(recs, global_pop, cfg.k), 'returned': len(recs)})
        # 召回对照：仅把热门配额翻倍，看覆盖率与热门占比如何变化。
        increased = dict(cfg.quotas()); increased['popular'] *= 2
        alternate, _, _ = engine.candidates(uid, quotas=increased)
        for label, cs in [('default', candidates), ('popular_x2', alternate)]:
            items = [c['item_id'] for c in cs]
            recalls.append({'user_id': uid, 'variant': label, 'length': len(items),
                            'recall': len(set(items) & actual) / len(actual),
                            'popular_share': len(set(items) & set(engine.ids[top_indices(engine.popularity, engine.ids, 100)])) / max(1, len(items))})
        rank_records.append({'user_id': uid, 'truth': sorted(actual), 'candidates': ids,
                             'rankings': rankings, 'cold_user': uid not in engine.histories})
        # 保存前两个用户的真实推荐列表作为案例（成功/失败案例分析素材）。
        if len(examples) < 2:
            examples.append({'user_id': uid, 'popular': [engine.enrich(i, scores['popular'][engine.pos[i]]) for i in rankings['popular']],
                             'itemcf': [engine.enrich(i, scores['itemcf'][engine.pos[i]]) for i in rankings['itemcf']]})
        if n % 100 == 0:
            print('evaluated users:', n, flush=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUTPUTS / 'per_user_metrics.csv', index=False)
    # 先按用户算指标再平均（与“合并所有预测再算”不同：每个用户权重相等）。
    table = detailed.groupby('strategy', sort=False).mean(numeric_only=True).reset_index()
    # AUC 例外：把全部用户候选合并计算（AUC 本质是全局排序质量的汇总统计）。
    table['auc'] = table.strategy.map({name: float(roc_auc_score(auc_y, pred)) if len(set(auc_y)) > 1 else None
                                       for name, pred in auc_preds.items()})
    table.to_csv(OUTPUTS / 'evaluation.csv', index=False)
    # 测试候选特征也落盘：同候选换模型/复跑实验时不用重算特征。
    pd.concat(samples, ignore_index=True).to_csv(PROCESSED / 'test_candidate_features.csv', index=False)
    recall_frame = pd.DataFrame(recalls)
    recall_frame.to_csv(OUTPUTS / 'recall_per_user.csv', index=False)
    recall_summary = recall_frame.groupby('variant').mean(numeric_only=True).reset_index()
    recall_summary.to_csv(OUTPUTS / 'recall_comparison.csv', index=False)
    pd.DataFrame([{'user_id': r['user_id'], 'items': json.dumps(r['candidates']), 'length': len(r['candidates'])}
                  for r in rank_records]).to_csv(PROCESSED / 'test_recall_set.csv', index=False)
    write_json(OUTPUTS / 'ranking_results.json', rank_records)
    write_json(OUTPUTS / 'cf_vs_popular_cases.json', examples)
    # 双模型 Top10 重叠率：衡量两个精排的输出是否趋同（高重叠≠高质量）。
    dual = float(np.mean([overlap(r['rankings']['lr'], r['rankings']['lightgbm'], cfg.k) for r in rank_records])) if 'lightgbm' in models else None
    summary = {'eligible_test_users': len(eligible), 'evaluated_users': len(users), 'k': cfg.k,
        'cold_users': sum(r['cold_user'] for r in rank_records),
        'future_unseen_truth_items': sum(len(set(r['truth']) - set(engine.ids)) for r in rank_records),
        'lr_lightgbm_overlap': dual,
        'metrics': json.loads(table.to_json(orient='records')), 'recall': json.loads(recall_summary.to_json(orient='records'))}
    write_json(OUTPUTS / 'evaluation.json', summary)
    # 留下简单、可重复生成的对比图；英文轴标签避免依赖系统中文字体。
    import matplotlib
    matplotlib.use('Agg')  # 无显示器的服务器环境也必须能出图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in zip(axes, ['precision_k', 'ndcg_k'], ['Precision@10', 'NDCG@10']):
        ax.bar(table.strategy, table[metric], color='#496b87')
        ax.set_title(title.replace('10', str(cfg.k)))
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUTS / 'model_comparison.png', dpi=160)
    plt.close(fig)
    write_report(summary, rank_records)
    # 所有报告写完后才标记可用；服务端据此拒绝读取半成品。
    write_json(OUTPUTS / 'run_status.json', {'stage': 'complete', 'complete': True})
    print(table.to_string(index=False), flush=True)


def write_report(summary, results):
    """根据落盘的评估结果自动生成 baseline_report.md（模块基线实验报告）。

    设计原则：报告内容全部来自 summary/训练日志/数据摘要，而不是自由文本；
    - 成功/失败案例从真实 ranking_results 里找：找不到就写“未发现该类型案例”，
      绝不编造一个“看起来合理的案例”；
    - LR 系数与 LightGBM 重要性读自落盘 CSV，配中文释义列出来；
    - 最后固定“结论与边界”段落，把标准化/负例含义/AUC 局限等说清楚，
      防止读者把抽样后的 predict_proba 误读成真实购买概率。
    """
    cfg = read_config()
    training = json.loads((OUTPUTS / 'training.json').read_text(encoding='utf-8'))
    data = json.loads((OUTPUTS / 'data_summary.json').read_text(encoding='utf-8'))
    lines = ['# 行业工程实践推荐系统模块基线报告', '', '类型：模块项目报告。个人信息由提交者填写。', '',
             '## 实现与数据', '',
             f"原始 {data['raw_rows']:,} 行，清洗后 {data['clean_rows']:,} 行；清洗顺序及删行数量见 stage01_retention.csv。",
             '按订单构建 Skip-gram 序列；使用历史期商品向量、二值余弦 ItemCF、热门和用户向量四路召回，去重后交给 StandardScaler + LogisticRegression 精排。',
             f"精排训练使用 {training['sample_rows']:,} 条样本，{training['positives']:,} 条正例。LightGBM：{training['lightgbm']}", '',
             '## 公平评估口径', '',
             f'历史期早于 {cfg.history_end}，精排标签期为 [{cfg.history_end}, {cfg.test_start})，最终测试期从 {cfg.test_start} 开始。所有用户、商品、向量及匹配特征冻结在历史期，标准化器只拟合精排训练样本。',
             '标签是对应未来窗口内、历史期未购买过的商品。候选只用历史数据产生；未召回和未见过的未来商品保留在 Recall 分母内，不强行补入候选。',
             f"测试使用种子 {cfg.seed} 从 {summary['eligible_test_users']} 个符合条件的用户抽取 {summary['evaluated_users']} 个，K={summary['k']}。用户列表已保存。",
             'AUC 在全部测试候选上计算，属于候选条件下的排序指标；Top-K 指标包含召回失败。LR 与 LightGBM 用相同候选、特征和标签。没有根据测试结果调参。', '',
             '| 策略 | Precision@K | Recall@K | NDCG@K | HitRate@K | AUC | 与同用户热门重叠率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['metrics']:
        auc = f"{r['auc']:.4f}" if r['auc'] is not None else '不适用'
        lines.append(f"| {r['strategy']} | {r['precision_k']:.4f} | {r['recall_k']:.4f} | {r['ndcg_k']:.4f} | {r['hit_rate_k']:.4f} | {auc} | {r['popular_overlap']:.4f} |")
    lines += ['', '## 召回与配额实验', '', '| 配额 | 平均长度 | 真实商品覆盖率 | 全局 Top100 热门占比 |', '|---|---:|---:|---:|']
    for r in summary['recall']:
        lines.append(f"| {r['variant']} | {r['length']:.2f} | {r['recall']:.4f} | {r['popular_share']:.4f} |")
    lines += ['', f'仅把热门配额从 {cfg.popular_quota} 增加到 {cfg.popular_quota*2}，总长度上限保持 {cfg.final_k}，其他配置不变。配额是上限，去重后不补满。',
              f"双模型 Top-K 重叠率：{summary['lr_lightgbm_overlap']}。重叠率越高表示输出越接近，不等于质量越高。", '', '## 成功与失败案例', '']
    for success in [True, False]:
        # 找真实案例：LR 是否命中了真实未来商品。找不到不编造。
        case = next((r for r in results if bool(set(r['truth']) & set(r['rankings']['lr'])) == success), None)
        if case is None:
            lines.append('本次抽样未发现' + ('命中' if success else '未命中') + '案例，不编造案例。')
            continue
        truth, candidates = set(case['truth']), set(case['candidates'])
        # 失败原因三分：召回失败 / 排序失败 / （命中则成功），由真实数据判定。
        reason = '候选和 LR Top-K 均覆盖了未来真实购买商品。' if success else (
            '候选没有覆盖未来真实商品，属于召回失败。' if not truth & candidates else '候选覆盖了部分真实商品，但 LR 未将其排入 Top-K，属于排序失败。')
        lines += [f"用户 {case['user_id']}：{reason}",
                  f"LR Top-K：{', '.join(case['rankings']['lr'])}；命中：{', '.join(sorted(truth & set(case['rankings']['lr']))) or '无'}。", '']
    coeff = pd.read_csv(OUTPUTS / 'lr_coefficients.csv').sort_values('standardized_coefficient', key=abs, ascending=False).head(3)
    lines += ['## 特征解释', '', 'LR 系数针对标准化后的特征，正值提高预测分、负值降低预测分，是关联关系。']
    meanings = {'user_log_orders': '用户历史订单活跃度', 'user_log_items': '历史购买商品种类数',
                'item_log_buyers': '商品历史购买人数', 'item_log_orders': '商品历史订单热度',
                'item_log_price': '历史平均单价', 'category_affinity': '用户历史类目偏好',
                'user_cosine': '用户与商品向量相似度', 'itemcf_score': '历史商品的 ItemCF 关联强度',
                'item2vec_score': '与最近历史商品的最大近邻分'}
    for r in coeff.itertuples():
        lines.append(f"- {r.feature}：系数 {r.standardized_coefficient:.4f}，表示{meanings.get(r.feature, '对应召回路是否命中')}。")
    if (OUTPUTS / 'lightgbm_importance.csv').exists() and 'lightgbm' in [r['strategy'] for r in summary['metrics']]:
        imp = pd.read_csv(OUTPUTS / 'lightgbm_importance.csv').head(3)
        lines += ['', 'LightGBM 最常用于分裂的三个特征：' + '；'.join(f"{r.feature}（{meanings.get(r.feature, '对应召回路是否命中')}，分裂次数 {r.importance}）" for r in imp.itertuples()) + '。分裂次数不表示正负影响。']
    lines += ['', '## 结论与边界', '',
        '这份基线贯通了清洗、订单序列、商品表示、向量检索、多路候选和监督精排。热门是必须保留的对照：复杂模型只有在同一用户集合和相同 K 下取得更好的命中与排序结果，才有证据说明增加的复杂度有价值。CF 与热门的重叠率用于检查个性化程度，不能单独证明推荐更好。',
        '当前历史特征冻结在历史窗口内，便于严格避免未来统计泄漏，但无法反映测试期新增兴趣；测试期新商品没有历史向量，只能计为召回失败。负样本来自候选中未在标签窗口购买的商品，未购买不等于不喜欢。训练按正负比例抽样，所以 predict_proba 只作为排序分数，不能解读为真实购买概率。AUC 是候选条件指标，应同时查看召回覆盖率、Precision、Recall 和 NDCG。后续可以在保持时间隔离的前提下增加滚动窗口训练、更新用户历史，并扩大测试用户范围。',
        '本次未加入探索模式、偏好设置或反馈闭环，这些不属于六阶段必做基线。具体示例的商品语义判断见 baseline/CASE_REVIEW.md，其中明确记录了审阅依据和主数据冲突。', '',
        '## 联调与演示', '', '运行 `python -m baseline.web`，默认访问 http://127.0.0.1:5002。使用 demo_ids.txt 的两个用户、两个商品。',
        '训练及推理模型均从 models/baseline 加载；测试覆盖时间隔离、已购过滤、候选配额、指标计算、模型保存加载及 HTTP 错误处理。测试执行结果见 tests_result.txt。']
    (OUTPUTS / 'baseline_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    """命令行入口：解析阶段参数并顺序执行。

    --eval-users：prepare/all 阶段生效，0 表示用全部合格用户评估（默认500）。
    用 threadpool_limits(limits=1) 包住全程：BLAS/OpenMP 单线程，
    避免 gensim/sklearn 多线程在重复运行时因线程数不同产生不可复现的数值。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--eval-users', type=int, default=500, help='prepare/all 时保存，0 表示全量评测')
    args = parser.parse_args()
    if args.eval_users < 0:
        parser.error('--eval-users 不能为负数')
    cfg = replace(Config(), eval_users=args.eval_users)
    with threadpool_limits(limits=1):
        if args.stage in ['prepare', 'all']:
            prepare(cfg)
        if args.stage in ['train', 'all']:
            train()
        if args.stage in ['evaluate', 'all']:
            evaluate()


if __name__ == '__main__':
    main()
