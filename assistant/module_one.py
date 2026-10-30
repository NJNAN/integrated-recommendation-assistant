"""复用模块一冻结产物：生成带证据的 GBP 目录，并在需要时拉起冻结服务。"""
import csv, json, math, os, re, socket, subprocess, sys, time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from config.settings import ROOT

COLORS = {'BLUE': '蓝色', 'RED': '红色', 'WHITE': '白色', 'BLACK': '黑色', 'GREEN': '绿色',
          'YELLOW': '黄色', 'PINK': '粉色', 'BROWN': '棕色', 'PURPLE': '紫色', 'ORANGE': '橙色'}
NOUNS = (('杯子', {'MUG', 'MUGS', 'CUP', 'CUPS'}), ('雨伞', {'UMBRELLA', 'UMBRELLAS'}),
         ('背包', {'BACKPACK', 'RUCKSACK'}))


def _prices(path, known):  # 读交易 CSV，聚合最后观测单价与交易次数
    latest, counts = {}, {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item, date = row.get('StockCode'), row.get('InvoiceDate', '')
            if item not in known or row.get('InvoiceNo', '').upper().startswith('C'):
                continue
            try:
                price, quantity = float(row['UnitPrice']), float(row['Quantity'])
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(price) and price > 0 and math.isfinite(quantity) and quantity > 0):
                continue
            if not re.match(r'^\d{4}-\d{2}-\d{2}', date):
                raise ValueError('历史价格需要 ISO 日期，不能按错误日期排序')
            counts[item] = counts.get(item, 0) + 1
            if item not in latest or date > latest[item]['date']:
                latest[item] = {'date': date, 'unit_price': price}
    return latest, counts


def import_catalog(catalog_path, transactions_path, output):
    """catalog.csv + train.csv -> data/module_one_catalog.json（保留价格、颜色与来源证据）。"""
    with Path(catalog_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    latest, counts = _prices(transactions_path, {row['item_id'] for row in rows})
    peak, items = max(counts.values(), default=1), []
    for row in rows:
        item_id, description = row['item_id'], row.get('description', '').strip()
        words = set(re.findall(r'[A-Z]+', description.upper()))
        colors = [zh for en, zh in COLORS.items() if en in words]
        category = next((name for name, nouns in NOUNS if words & nouns), row.get('category') or '未分类')
        observed = latest.get(item_id)
        items.append({
            'item_id': item_id, 'name': row.get('name') or description or item_id, 'category': category,
            'original_category': row.get('category'), 'description': description, 'currency': 'GBP',
            'color': colors[0] if len(colors) == 1 else ('多色' if colors else '未标注'),
            'price': observed['unit_price'] if observed else None,
            'price_kind': 'historical_last_observed' if observed else 'missing',
            'price_source': {'file': 'development/data/train.csv', 'date': observed['date'],
                             'field': 'UnitPrice', 'currency': 'GBP'} if observed else None,
            'color_source': {'field': 'description', 'tokens': sorted(words & set(COLORS))},
            'popularity': round(counts.get(item_id, 0) / peak, 6),
            'tags': [category, row.get('category') or '未分类'], 'visual_labels': [], })
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
    """在 127.0.0.1 拉起模块一冻结包，日志与数据库都写进本项目 runtime。"""

    def __init__(self, settings):
        self.settings, self.process, self.log = settings, None, None

    def start(self):
        with socket.socket() as probe:  # 先探测端口，避免子进程静默失败
            try:
                probe.bind(('127.0.0.1', self.settings.module_one_port))
            except OSError as exc:
                raise RuntimeError('模块一端口已占用，请设置 IRA_MODULE_ONE_PORT') from exc
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
            self.close()
            raise

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log and self.log.close()


def integrated_settings(settings):
    """入口自动接入模块一；demo、显式配置或缺文件时返回 (settings, None) 保持离线。"""
    if os.getenv('IRA_PROFILE', 'auto') == 'demo' or settings.recommender_url or os.getenv('IRA_CATALOG'):
        return settings, None
    root = settings.module_one_root
    files = (root / 'release/web_system/data/catalog.csv', root / 'development/data/train.csv',
             root / 'release/expected/submission_v2_R1_head48.csv')
    if not all(path.is_file() for path in files):
        return settings, None
    import_catalog(files[0], files[1], output := ROOT / 'data/module_one_catalog.json')
    active = replace(settings, catalog_path=output, recommender_accounts=settings.module_one_accounts,
                     recommender_url=f'http://127.0.0.1:{settings.module_one_port}',
                     upstream_timeout_seconds=5)
    return active, ModuleOneService(active)
