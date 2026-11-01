"""复用模块一冻结产物：生成带证据的 GBP 目录，并在需要时拉起冻结服务。

讲解顺序：
1. `_prices`           交易 CSV → 每件商品“最后观测单价 + 交易次数”。
2. `import_catalog`   商品表 + 价格 → data/module_one_catalog.json（每件商品都带来源证据）。
3. `ModuleOneService` 拉起 / 关闭模块一的冻结 Flask 服务（端口 5001）。
4. `integrated_settings` 决定“接真实服务”还是“走离线”，并据此换掉 settings。
"""
import csv, json, math, os, re, socket, subprocess, sys, time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from config.settings import ROOT

# 英文颜色词 → 中文颜色：只认真实出现在描述里的颜色词，取不到就标“未标注”，绝不猜。
COLORS = {'BLUE': '蓝色', 'RED': '红色', 'WHITE': '白色', 'BLACK': '黑色', 'GREEN': '绿色',
          'YELLOW': '黄色', 'PINK': '粉色', 'BROWN': '棕色', 'PURPLE': '紫色', 'ORANGE': '橙色'}
# 商品名词 → 品类：真实目录的 category 字段比较粗，用描述里的名词补出“杯子/雨伞/背包”这些可推荐品类。
NOUNS = (('杯子', {'MUG', 'MUGS', 'CUP', 'CUPS'}), ('雨伞', {'UMBRELLA', 'UMBRELLAS'}),
         ('背包', {'BACKPACK', 'RUCKSACK'}))


def _prices(path, known):  # 读交易 CSV，聚合最后观测单价与交易次数
    """返回 (最后观测价格表, 交易次数表)。

    只统计“正向交易”：退货单（InvoiceNo 以 C 开头）、非正数价格/数量都跳过；
    日期必须是 ISO 格式，否则直接报错——宁可失败也不能按错误日期排序。
    """
    latest, counts = {}, {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item, date = row.get('StockCode'), row.get('InvoiceDate', '')
            if item not in known or row.get('InvoiceNo', '').upper().startswith('C'):
                continue                                            # 不在商品表里，或是退货单
            try:
                price, quantity = float(row['UnitPrice']), float(row['Quantity'])
            except (KeyError, ValueError):
                continue                                            # 脏行直接跳过
            if not (math.isfinite(price) and price > 0 and math.isfinite(quantity) and quantity > 0):
                continue                                            # 非正数/非有限值不算成交
            if not re.match(r'^\d{4}-\d{2}-\d{2}', date):
                raise ValueError('历史价格需要 ISO 日期，不能按错误日期排序')
            counts[item] = counts.get(item, 0) + 1                   # 交易次数 → popularity
            if item not in latest or date > latest[item]['date']:
                latest[item] = {'date': date, 'unit_price': price}   # 只保留“最后一次观测”的价格
    return latest, counts


def import_catalog(catalog_path, transactions_path, output):
    """catalog.csv + train.csv → data/module_one_catalog.json（保留价格、颜色与来源证据）。"""
    with Path(catalog_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    latest, counts = _prices(transactions_path, {row['item_id'] for row in rows})
    peak, items = max(counts.values(), default=1), []
    for row in rows:
        item_id, description = row['item_id'], row.get('description', '').strip()
        words = set(re.findall(r'[A-Z]+', description.upper()))      # 描述里的英文单词（用于颜色/品类推断）

        # 颜色：只认唯一出现的颜色词；出现多个标“多色”，一个都没有标“未标注”。
        colors = [zh for en, zh in COLORS.items() if en in words]
        # 品类：先用描述里的名词覆盖（如 MUG → 杯子），否则用商品表原品类。
        category = next((name for name, nouns in NOUNS if words & nouns), row.get('category') or '未分类')
        observed = latest.get(item_id)
        items.append({
            'item_id': item_id, 'name': row.get('name') or description or item_id, 'category': category,
            'original_category': row.get('category'), 'description': description, 'currency': 'GBP',
            'color': colors[0] if len(colors) == 1 else ('多色' if colors else '未标注'),
            # 价格：没有成交记录就是 None，并且必须用 price_kind 明确声明“缺失”。
            'price': observed['unit_price'] if observed else None,
            'price_kind': 'historical_last_observed' if observed else 'missing',
            # 价格来源：文件/日期/字段/币种，解释和追溯都靠它。
            'price_source': {'file': 'development/data/train.csv', 'date': observed['date'],
                             'field': 'UnitPrice', 'currency': 'GBP'} if observed else None,
            # 颜色来源：来自哪个字段、命中了哪些英文词。
            'color_source': {'field': 'description', 'tokens': sorted(words & set(COLORS))},
            # 热度归一化到 0~1：本商品交易次数 / 全目录峰值次数。
            'popularity': round(counts.get(item_id, 0) / peak, 6),
            'tags': [category, row.get('category') or '未分类'], 'visual_labels': [], })

    # 文档级元信息：数据集名/来源类型/币种/模型版本/价格口径说明/数据引用/源文件哈希。
    document = {'dataset': 'online-retail-module-one-v2', 'source_kind': 'module_one_real',
                'currency': 'GBP', 'model_version': 'Oblivionis(v2-r1-head48)',
                'price_notice': '历史交易最后观测单价，不是当前售价；来源单位英镑，不自动换算人民币',
                'dataset_reference': 'https://archive.ics.uci.edu/dataset/502/online+retail+ii',
                'sources': [{'name': Path(p).name, 'sha256': sha256(Path(p).read_bytes()).hexdigest()}
                            for p in (catalog_path, transactions_path)], 'items': items}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
                            encoding='utf-8')
    return document


class ModuleOneService:
    """在 127.0.0.1 拉起模块一的冻结包，日志与数据库都写进本项目 runtime/。"""

    def __init__(self, settings):
        self.settings, self.process, self.log = settings, None, None

    def start(self):
        # 先探测端口：被占用时立刻给出明确错误，而不是让子进程静默失败。
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', self.settings.module_one_port))
            except OSError as exc:
                raise RuntimeError('模块一端口已占用，请设置 IRA_MODULE_ONE_PORT') from exc

        # 子进程跑 integrations/module_one_server.py（它只 import 模块一 release 的公开 app）。
        runtime, command = ROOT / 'runtime/module_one', [sys.executable, '-B', '-u',
                         str(ROOT / 'integrations/module_one_server.py'),
                         str(self.settings.module_one_root / 'release'), str(self.settings.module_one_port)]
        runtime.mkdir(parents=True, exist_ok=True)
        self.log = (runtime / 'server.log').open('ab')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                   OBLIVIONIS_DATABASE_PATH=str(runtime / 'serving.sqlite3'))
        self.process = subprocess.Popen(command, cwd=runtime, env=env, stdout=self.log, stderr=self.log,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            # 轮询健康检查：只有 model_ready 为真才算就绪（最多等 20 秒）。
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('模块一服务未启动，请查看 runtime/module_one/server.log')
                try:
                    with urlopen(self.settings.recommender_url + '/api/health', timeout=1) as response:
                        if json.load(response)['data']['model_ready']:
                            return
                except (OSError, ValueError, KeyError):
                    time.sleep(0.2)
            raise RuntimeError('模块一服务启动超时')
        except Exception:
            self.close()                                            # 启动失败要回收子进程，不留僵尸
            raise

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()                                # 先礼后兵：terminate，5 秒不退再 kill
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log and self.log.close()


def integrated_settings(settings):
    """入口自动接入模块一；demo、显式配置或缺文件时返回 (settings, None) 保持离线。"""
    # 三种情况不接真实服务：显式 demo、显式给了上游/目录、用户自己指定了目录。
    if os.getenv('IRA_PROFILE', 'auto') == 'demo' or settings.recommender_url or os.getenv('IRA_CATALOG'):
        return settings, None
    root = settings.module_one_root
    release = root / 'release'
    catalog_csv = release / 'web_system/data/catalog.csv'
    transactions = root / 'development/data/train.csv'
    scores = release / 'expected/submission_v2_R1_head48.csv'
    output = ROOT / 'data/module_one_catalog.json'
    # 发布包（web_system 源码 + 冻结排序 + 商品目录）必须存在，否则视为没有模块一。
    if not (catalog_csv.is_file() and scores.is_file()):
        return settings, None                                       # 没有模块一代码：纯离线
    # 有原始交易数据就现场重算目录；仓库里的 vendor 副本不含 72MB 原始数据，
    # 此时直接用已生成的 data/module_one_catalog.json（内容与原始数据一致，带 sha256 来源）。
    if transactions.is_file():
        import_catalog(catalog_csv, transactions, output)
    elif not output.is_file():
        return settings, None
    # 再把 settings 指向本机 5001 的服务与这份目录。
    active = replace(settings, catalog_path=output, recommender_accounts=settings.module_one_accounts,
                     recommender_url=f'http://127.0.0.1:{settings.module_one_port}',
                     upstream_timeout_seconds=5)
    return active, ModuleOneService(active)
