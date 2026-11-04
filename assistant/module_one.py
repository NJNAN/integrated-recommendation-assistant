"""复用模块一冻结产物：生成带证据的 GBP 目录，并在需要时拉起冻结服务。

讲解顺序：
1. `_prices`           交易 CSV → 每件商品“最后观测单价 + 交易次数”。
2. `import_catalog`   商品表 + 价格 → data/module_one_catalog.json（每件商品都带来源证据）。
3. `ModuleOneService` 拉起 / 关闭模块一的冻结 Flask 服务（端口 5001）。
4. `integrated_settings` 决定“接真实服务”还是“走离线”，并据此换掉 settings。

阅读约定：每个函数的 docstring 都写了「参数（每个参数是什么、从哪来、取值）」「返回」「异常」。
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
    """读取历史交易 CSV，过滤退货和无效交易，然后整理每件商品最后一次观测到的交易单价，并统计有效交易次数。


    参数：
        path (str | pathlib.Path): 交易明细 CSV 的路径，来自模块一项目的
            `development/data/train.csv`（UCI Online Retail II 的原始交易表）。
            必需列：StockCode（商品ID）、InvoiceNo（订单号）、InvoiceDate（时间）、
            UnitPrice（单价，英镑）、Quantity（数量）。
        known (set[str]): 只统计这些商品 ID（就是商品表 catalog.csv 里有的 ID）。
            交易表里出现但商品表里没有的 ID 会被跳过，避免生成“没有名字”的孤儿商品。

    返回：
        tuple，两个值：
        1. `latest` (dict): {商品ID: {'date': 'YYYY-MM-DD…', 'unit_price': float}}，
           每件商品**最后一次**正向交易的日期和单价（不是平均价，也不是现价）。
        2. `counts` (dict): {商品ID: 交易次数}，后面用来算 popularity（热度）。

    过滤规则（都是为了不把脏数据当价格）：
        - 退货单（InvoiceNo 以 C 开头）跳过；
        - 价格或数量不是正数、不是有限数（NaN/inf）的跳过；
        - 日期格式不是 ISO（YYYY-MM-DD 开头）→ 直接抛错，宁可失败也不按错误日期排序。

    异常：
        ValueError: 日期不是 ISO 格式时抛出（调用方会把它当成目录生成失败）。
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
    """把商品表和交易数据合并成统一的 JSON 商品目录。这里主要处理价格、颜色、品类和商品热度。
价格使用历史交易单价，缺少价格就明确标记；颜色和品类根据商品描述提取，无法确定时不会随意猜测；热
度则根据有效交易次数归一化得到。同时，文件还会记录数据来源和哈希值，方便追踪。

    参数：
        catalog_path (str | pathlib.Path): 模块一发布包里的商品表
            `release/web_system/data/catalog.csv`。
            必需列：item_id、name（可空）、category（可空）、description（英文描述）。
        transactions_path (str | pathlib.Path): 模块一的历史交易表
            `development/data/train.csv`（见 `_prices` 的参数说明）。
        output (str | pathlib.Path): 生成的 JSON 路径，正常是
            `data/module_one_catalog.json`。父目录不存在会自动创建。

    返回：
        dict: 生成的整个目录文档（同时已写入 output 文件），结构：
        - dataset / source_kind / currency / model_version / price_notice / dataset_reference；
        - sources：两个源文件的文件名 + sha256（数据来源可追溯）；
        - items：商品数组，每件商品包含 item_id/name/category/color/price/price_kind/
          price_source/color_source/popularity/tags/description/visual_labels。

    生成的规则（答辩要点）：
        - 价格：取最后观测单价，单位 GBP，`price_kind='historical_last_observed'`；
          没有成交记录的商品 `price=None` 且 `price_kind='missing'`；
        - 颜色：只从英文描述里找颜色词，唯一才算（多个→“多色”，没有→“未标注”），不猜；
        - 品类：先用描述里的名词覆盖（MUG/CUP→杯子、UMBRELLA→雨伞、BACKPACK→背包），否则用原品类；
        - 热度：交易次数 / 全目录峰值次数，归一化到 0~1。

    异常：
        ValueError: 传入的 CSV 不合法时（由 `_prices` 或 CSV 解析抛出）。
        OSError: 文件读不了或输出目录不能创建。
    """
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
        """只记下配置，不启动任何东西（进程在 `start()` 里才起）。

        参数：
            settings (config.settings.Settings): 本类用到三项：
                - `module_one_port` (int)：模块一服务端口，默认 5001；
                - `module_one_root` (Path)：模块一代码目录（同级项目或 vendor 副本），
                  其下的 `release/` 就是要跑的服务包；
                - `recommender_url` (str)：启动后用来探活的地址，形如 http://127.0.0.1:5001。

        副作用：
            `self.process`（子进程句柄，初始 None）、`self.log`（日志文件句柄，初始 None）。
        """
        self.settings, self.process, self.log = settings, None, None

    def start(self):
        """负责启动模块一的服务，并通过健康检查确认服务是否准备完成。启动失败时也会清理进程。

        参数：
            无（用 `__init__` 里记下的 settings）。

        返回：
            None。成功时返回即表示 `/api/health` 回包中 `data.model_ready` 为 True，
            后续 HTTPRecommender 可以开始登录和取推荐了。

        副作用：
            - 在 `runtime/module_one/` 下创建日志文件 `server.log` 和运行期数据库 `serving.sqlite3`；
            - 起一个子进程：`python -B integrations/module_one_server.py <release目录> <端口>`；
            - 把子进程句柄和日志句柄存到 `self.process` / `self.log`，供 `close()` 回收。

        异常：
            RuntimeError: 端口被占用 / 子进程没起来 / 20 秒内没就绪（三种都会带中文提示）。
            异常前会自动调用 `close()`，不会留下僵尸进程。
        """
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
        """关闭模块一子进程并关掉日志（可以重复调用，没启动时也安全）。

        参数：
            无。

        返回：
            None。

        做法：
            先 `terminate()` 客气地请它退出，等 5 秒还不退就 `kill()`；
            最后关闭日志文件。这样程序退出时不会留下 Flask 僵尸进程。
        """
        if self.process and self.process.poll() is None:
            self.process.terminate()                                # 先礼后兵：terminate，5 秒不退再 kill
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log and self.log.close()


def integrated_settings(settings):
    """根据当前环境决定连接真实推荐服务，还是使用离线商品目录。
真正的 HTTP 请求封装在 `adapters.py` 中，通过接口调用模块一，不需要直接修改模块一的内部实现。
所以，这个文件主要解决的是两个模块之间的数据转换和服务对接问题。

    参数：
        settings (config.settings.Settings): 原始配置。本函数读：
            - `module_one_root` (Path)：模块一代码目录（同级项目或仓库 vendor 副本）；
            - `module_one_port` / `module_one_accounts`：启动服务要用的端口和演示账号；
            - `recommender_url`（若已显式配置，则不再自动接入）。

    返回：
        tuple，两个值：
        1. `active` (Settings)：实际要用的配置。接真实服务时会把
           `catalog_path` 换成生成的真实目录、`recommender_url` 指向本机 5001、
           `upstream_timeout_seconds` 放宽到 5 秒；走离线时原样返回传入的 settings。
        2. `service` (ModuleOneService | None)：需要启动的服务对象；离线时是 None，
           调用方（app.py）决定是否 `.start()`。

    三种走离线的情况（直接返回 (settings, None)，不启服务）：
        - 显式 `IRA_PROFILE=demo`；
        - 显式配置了 `IRA_RECOMMENDER_URL` 或 `IRA_CATALOG`（用户自己指定，不自动接入）；
        - 找不到模块一发布包（`release/web_system/data/catalog.csv` 或冻结排序结果不存在）。

    数据来源的两种情况：
        - 有原始交易数据 `development/data/train.csv`：现场重算 `data/module_one_catalog.json`；
        - 只有 vendor 副本（没有 72MB 原始数据）：直接用仓库里已生成的目录文件。

    异常：
        一般不抛异常；`import_catalog` 内部出错会向上抛（由 app.py 捕获并回退离线演示）。
    """
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
