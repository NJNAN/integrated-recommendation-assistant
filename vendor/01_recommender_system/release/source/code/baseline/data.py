"""清洗、按订单切分时间窗口及订单序列构建（数据层）。

在流水线中的位置（第 1 阶段 prepare 的核心实现）：
    输入：RAW/transactions.csv（Online Retail B2B 批发数据集原始交易，约 39.8 万行）
          与 RAW/products.csv（商品主数据：编码、英文描述、中文名、类目）。
    处理：三级清洗漏斗（任务书规定的删除顺序）→ 统一重命名列 → 与商品主数据
          左连接补全展示信息 → 时间切分成 history / rank 标签期 / test 三窗口 →
          把 history 期交易压缩成"订单 = 一个商品序列"的形式。
    输出：交给 pipeline.prepare() 落盘为 clean_transactions.csv、catalog.csv、
          history.csv、rank_train.csv、test.csv、user_sequences.csv 等。

被谁调用 / 调用谁：
    - pipeline.prepare() 调用 clean_data / split_data / sequences / write_json；
    - tests/test_baseline.py 直接调用本模块验证清洗漏斗顺序与切分边界行为；
    - 本模块只依赖 pandas / json / pathlib，不依赖项目内其他模块（配置通过参数
      cfg 传入，避免循环依赖）。

关键概念：
    - 清洗漏斗（funnel）：按固定顺序逐步删除非法行，并记录每一步删了多少、
      还剩多少（保留率 retention），全过程可审计、可复现；
    - 时间切分防泄漏（temporal split, no leakage）：模型只能看到"过去"的数据。
      若用全量数据（含未来）统计商品热度、训练向量，离线评估就会虚高——相当于
      考试前拿到了答案。这里以订单为单位整体划入某一个时间窗口，防止同一张订单
      的行被拆到窗口两侧造成信息穿越。
"""
import json
from pathlib import Path

import pandas as pd


def write_json(path, value):
    """把 Python 对象序列化为 JSON 并写入文件（自动创建父目录）。

    参数：
        path  : 目标文件路径（str 或 Path），父目录不存在时自动 mkdir -p；
        value : 任意可 JSON 序列化的对象（dict / list / 标量）。

    设计说明：
        - ensure_ascii=False：中文原样写入（而不是 \\uXXXX 转义），报告可读；
        - allow_nan=False：NaN/Infinity 不是合法 JSON，一旦出现直接抛错而不是
          写出 "NaN" 这种会让下游 json.loads 失败的内容——宁可早失败；
        - indent=2：缩进两格，人工查看方便。
    被 pipeline.prepare / train / evaluate 大量调用，是全项目统一的 JSON 落盘出口。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def clean_data(tx, products):
    """清洗原始交易表并与商品主数据合并，产出可直接建模的干净数据。

    参数：
        tx       : 原始交易 DataFrame，必须含 InvoiceNo / StockCode / CustomerID /
                   InvoiceDate / Quantity / UnitPrice / Country 七列；
        products : 商品主数据 DataFrame，含 StockCode / Description /
                   Chinese_Description / Product_Category 列。

    返回（四元组）：
        tx        : 清洗后的交易表（列已重命名为 user_id / item_id / invoice_no /
                    transaction_time，并附加 name / description / category /
                    catalog_match / source_row_no）；
        catalog   : 商品目录表（主数据匹配到的 + 未匹配商品的占位行）；
        stages    : 清洗漏斗各步骤统计 DataFrame（stage / before / removed /
                    after / retention），用于报告"每一步删了多少行"；
        rejected  : 所有被删除的行合集（带 reason 列标注删除原因），可审计。

    设计理由：
        1. 删除顺序遵循任务书规定的漏斗顺序（missing_user → return_or_cancel →
           invalid_price → invalid_quantity → missing_key → invalid_date →
           duplicate）。顺序影响每一步的删除计数，因此顺序本身就是契约，
           测试 test_cleaning_funnel_and_duplicate 会校验；
        2. 每一行被删都留下记录（rejected），"删了什么、为什么删"完全透明；
        3. 未匹配主数据的商品【不剔除】——它们仍是合法交易，只是缺展示信息，
           用占位名"未知商品"补齐（避免为凑主数据而丢弃真实购买行为）；
        4. source_row_no 记录原始文件中的行号（Excel/CSV 第 2 行起为数据行，
           第 1 行是表头），排序时作为最终 tie-break，保证清洗结果确定可复现。
    """
    # copy()：不修改调用方传入的原表（防御性拷贝），函数无副作用。
    tx = tx.copy()
    # 必需字段集合。集合差集 required - set(tx) 得到"缺哪些列"，缺列直接报错，
    # 比后面 KeyError 更早、更明确地指出输入不合法。
    required = {'InvoiceNo', 'StockCode', 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country'}
    if required - set(tx):
        raise ValueError(f'交易表缺字段: {sorted(required - set(tx))}')
    # 记住原始列清单：后面判断"完全重复行"时以原始列为准（不含清洗中新增的列）。
    raw_columns = list(tx.columns)
    # 三个 ID 类字段统一转成 pandas 的 string 扩展类型并去掉首尾空白。
    # 为什么用 'string'（StringDtype）而不是 object：object 列里空值是浮点 NaN，
    # 与字符串混型易出错；StringDtype 的空值是可判定的 pd.NA，缺失语义清晰。
    # 空字符串 '' 视为缺失（replace('', pd.NA)）：原始数据里"看起来有值其实是空串"
    # 的 ID 等同于没填。
    for col in ['InvoiceNo', 'StockCode', 'CustomerID']:
        tx[col] = tx[col].astype('string').str.strip().replace('', pd.NA)
    # 原始 CSV 中 CustomerID 有 "12345.0" 这种带小数点尾巴的写法（读表时被当成
    # 浮点数再转字符串所致），正则 \.0$ 把结尾的 ".0" 去掉，还原成纯数字字符串。
    # r'\.0$' 是原始字符串：\. 在正则里表示"字面点号"（不转义的话 . 匹配任意字符），
    # $ 表示行尾——即只删除恰好以 ".0" 结尾的部分，不会误伤 "1230" 这类正常 ID。
    tx['CustomerID'] = tx.CustomerID.str.replace(r'\.0$', '', regex=True)
    # 数量与单价转数值型。errors='coerce'：无法解析的脏值（如空串、字母）不抛异常，
    # 而是置为 NaN，留给后续 invalid_price / invalid_quantity 步骤统一删除。
    for col in ['Quantity', 'UnitPrice']:
        tx[col] = pd.to_numeric(tx[col], errors='coerce')
    # source_row_no：原始文件行号（range 从 2 开始，因为第 1 行是表头）。
    # 作用：任何"平局"排序（同时间、同订单、同商品）都以原始文件出现顺序为准，
    # 使清洗结果与运行环境无关、完全可复现。
    tx['source_row_no'] = range(2, len(tx) + 2)
    raw_count = len(tx)          # 清洗前总行数，作为保留率的分母。
    stages, rejected = [], []    # 漏斗统计记录 / 被删除行收集器。

    def drop(reason, mask):
        """执行一步漏斗删除：把 mask 为 True 的行移出 tx 并记账。

        参数：
            reason : 本步骤名称（会写进 rejected 的 reason 列与 stages 的 stage 列）；
            mask   : 布尔 Series，True 表示该行应被删除。
        通过 nonlocal 修改外层 tx（闭包写法），使七步漏斗共享同一份递进状态。
        fillna(False)：mask 中若有 pd.NA（如 isna 比较产生的缺失），保守视为
        "不删除"，避免 NA 传播导致整列删除行为不确定。
        """
        nonlocal tx
        mask = mask.fillna(False)
        bad = tx.loc[mask].copy()          # 将被删的行复制出来留档
        bad['reason'] = reason             # 标注删除原因
        rejected.append(bad)
        before = len(tx)
        tx = tx.loc[~mask].copy()          # ~ 按位取反：保留 mask 为 False 的行
        # retention 用 max(1, raw_count) 防止除零（理论上 raw_count>0，防御性写法）。
        stages.append({'stage': reason, 'before': before, 'removed': len(bad),
                       'after': len(tx), 'retention': len(tx) / max(1, raw_count)})

    # 任务书要求的三步漏斗顺序。（后续步骤同样按任务书规定的固定顺序执行，
    # 顺序即契约：同一份数据在任何机器上跑出的各步删除计数必须一致。）
    # 第 1 步 missing_user：没有 CustomerID 的行无法归属到任何用户，
    # 而"用户历史购买序列"是本流水线一切个性化的基础，匿名交易直接删除。
    drop('missing_user', tx.CustomerID.isna())
    # 第 2 步 return_or_cancel：退货/取消行。Online Retail 数据集中发票号以 "C"
    # 开头（Credit note，红字冲销单）表示取消/退货；Quantity<=0 同样表示退回。
    # 推荐系统建模的是"用户想买什么"，退货不代表购买意图，必须剔除，
    # 否则会把退掉的商品当成用户的正反馈。str.upper() 兼容小写 'c' 开头；
    # na=False 表示 InvoiceNo 缺失的行在这一步不算命中（它会在 missing_key 步删除）。
    drop('return_or_cancel', tx.InvoiceNo.str.upper().str.startswith('C', na=False) | tx.Quantity.le(0))
    # 第 3 步 invalid_price：单价缺失或 <=0。0 价通常是赠品/调整项，负价是退款残留；
    # 价格后面要参与精排特征（item_log_price），非法价格会污染特征分布。
    drop('invalid_price', tx.UnitPrice.isna() | tx.UnitPrice.le(0))
    # 第 4 步 invalid_quantity：数量缺失。此时 <=0 的已在第 2 步删除，剩下的 NaN
    # 无法确认是否为正数购买，保守删除。
    drop('invalid_quantity', tx.Quantity.isna())
    # 第 5 步 missing_key：发票号或商品编码缺失。二者是"订单序列"的主键——
    # 缺任何一个就无法把行归入某次购买的某个商品，序列构建无从谈起。
    drop('missing_key', tx.InvoiceNo.isna() | tx.StockCode.isna())
    # 解析交易时间。format='mixed'：原始数据日期格式混杂（如 "12/1/2010 14:23" 与
    # ISO 格式并存），让 pandas 逐行自动推断格式；dayfirst=False 表示按美式
    # 月/日/年解读（数据集来自英国公司但导出格式为美式）；errors='coerce' 把
    # 解析失败置为 NaT（Not a Time），交给下一步删除。
    tx['transaction_time'] = pd.to_datetime(tx.InvoiceDate, format='mixed', dayfirst=False, errors='coerce')
    # 第 6 步 invalid_date：时间无法解析的行。没有时间戳就无法参与时间切分，删除。
    drop('invalid_date', tx.transaction_time.isna())
    # 第 7 步 duplicate：完全重复行（所有原始列都相同）。keep='first' 保留第一次
    # 出现。重复行会虚增热度与购买频次统计；注意只删"整行完全一样"的重复，
    # 同一订单买同一商品两行若数量/价格不同则是两条合法记录，不删。
    drop('duplicate', tx.duplicated(subset=raw_columns, keep='first'))
    # 重命名为流水线内部统一的小写蛇形命名：CustomerID->user_id、
    # StockCode->item_id、InvoiceNo->invoice_no。之后所有模块只认这套名字，
    # 与原始数据的列名解耦。
    tx = tx.rename(columns={'CustomerID': 'user_id', 'StockCode': 'item_id', 'InvoiceNo': 'invoice_no'})
    # ---- 商品主数据（catalog）处理 ----
    products = products.copy()
    # 主数据的 StockCode 同样转 string 并去空白，保证与交易表 item_id 的连接键
    # 类型一致（一边是 object 一边是 StringDtype 时 merge 会匹配不上）。
    products['StockCode'] = products.StockCode.astype('string').str.strip()
    # 去掉编码缺失的行，并按 (StockCode, Description) 稳定排序（kind='stable'
    # 保证相同键的行保持原有相对顺序，排序结果确定）。
    products = products.dropna(subset=['StockCode']).sort_values(['StockCode', 'Description'], kind='stable')
    # 同一编码可能有多行（不同描述版本）：先排序再 drop_duplicates('StockCode')
    # 保留排序后的第一条，使"每个商品编码恰好一行描述"，满足 merge 的
    # many_to_one 约束。同时重命名列：英文描述 -> description，中文描述 -> name
    # （前端展示优先用中文名），类目 -> category。
    catalog = products.drop_duplicates('StockCode').rename(columns={
        'StockCode': 'item_id', 'Description': 'description',
        'Chinese_Description': 'name', 'Product_Category': 'category'})
    # 只保留四列并填空串：展示字段缺失不应是 NaN（NaN 会传染进 JSON/CSV 比较），
    # 统一用 '' 表示"无内容"。
    catalog = catalog[['item_id', 'description', 'name', 'category']].fillna('')
    # 左连接把商品信息挂到每条交易上。validate='many_to_one'：显式声明"多条交易
    # 对一个商品"的关系，若右表出现重复 item_id 会直接报错（数据契约检查）。
    # indicator=True 会生成 _merge 列标记每行是 both（匹配上）还是 left_only
    # （未匹配），据此得到 catalog_match 布尔列——后续报告需要统计未匹配比例。
    tx = tx.merge(catalog, on='item_id', how='left', validate='many_to_one', indicator=True)
    tx['catalog_match'] = tx['_merge'].eq('both')
    tx = tx.drop(columns=['_merge'])          # _merge 是临时标记列，用完即删
    # 连接产生的 NaN（未匹配行）统一填成空串，与 catalog 的空值口径一致。
    for col in ['name', 'description', 'category']:
        tx[col] = tx[col].fillna('')
    # 不按低频或主数据缺失剔除合法交易。未匹配商品也有确定性展示信息。
    # 类目为空 → '未知'；名称为空 → 退回英文描述；描述也为空 → '未知商品'。
    # 三层兜底保证每条交易都有可展示的名称与类目，前端不会出现空白/NaN。
    tx.loc[tx['category'].eq(''), 'category'] = '未知'
    tx.loc[tx['name'].eq(''), 'name'] = tx.loc[tx['name'].eq(''), 'description']
    tx.loc[tx['name'].eq(''), 'name'] = '未知商品'
    # 未匹配商品不读取未来交易的名称：统一使用占位名。
    # 把这些商品也追加进 catalog（description='' / name='未知商品' / category='未知'），
    # 使 catalog 覆盖交易中出现过的【全部】商品编码——engine 构建索引、web 展示
    # 时按 item_id 查 catalog 永远不会缺键。之所以坚持用占位名而不是想办法补全：
    # 补全只能来自未来数据或外部猜测，前者是时间泄漏，后者不可验证。
    extra = tx.loc[~tx.catalog_match, ['item_id']].drop_duplicates()
    extra = extra.assign(description='', name='未知商品', category='未知')
    # concat 合并后按 item_id 排序、重置索引：catalog 的行顺序确定，
    # 两次运行产出逐字节一致的文件（可复现性）。
    catalog = pd.concat([catalog, extra], ignore_index=True).sort_values('item_id').reset_index(drop=True)
    # 全表最终排序：时间 → 订单号 → 商品 → 原始行号，四级排序全部稳定，
    # 输出行顺序完全确定。这一步同时保证后续 groupby / 序列构建看到的
    # 数据天然有序，减少各处的隐式排序依赖。
    tx = tx.sort_values(['transaction_time', 'invoice_no', 'item_id', 'source_row_no'], kind='stable').reset_index(drop=True)
    return tx, catalog, pd.DataFrame(stages), pd.concat(rejected, ignore_index=True)


def split_data(clean, cfg):
    """按时间把清洗后的交易切成三个互不重叠的窗口（防泄漏切分）。

    参数：
        clean : clean_data 的输出交易表（须含 user_id / invoice_no / transaction_time）；
        cfg   : Config 对象，使用其 history_end 与 test_start 两个时间边界。

    返回：
        长度为 3 的列表 [history, rank_train, test]：
        - history   : transaction_time < history_end，建模期——训练 Item2Vec、
                      统计热度/ItemCF/用户历史，一切商品表示冻结于此；
        - rank_train: [history_end, test_start)，精排标签期——产生 LR/LightGBM
                      的训练正负样本；
        - test      : >= test_start，最终测试期——只用于评估，绝不参与训练。

    为什么必须按时间切而不是随机切：
        推荐系统上线后永远只能拿"过去"预测"未来"。随机切分会让训练集里混入
        测试期的购买记录（例如同一用户 9 月的购买进了训练集、10 月的进了测试集，
        而模型统计热度时把 9 月也算进去了），离线指标虚高、上线暴跌——
        这就是"未来数据泄漏"。时间切分保证评估模拟的是真实上线场景。

    为什么以订单（而非单行）为单位切分：
        B2B 批发场景一张订单可能跨午夜录入，个别行时间戳落在边界两侧。若按行切，
        同一订单的一半进 history、一半进 test，模型在训练时"见过"这张订单，
        测试时又拿它当未来购买，构成订单级泄漏。这里先按 (user_id, invoice_no)
        分组取组内最大时间（transform('max') 把组级结果广播回每一行），
        整张订单按其最晚时间戳归入同一个窗口。

    空窗口检查：任何一个窗口为空说明时间边界设得不合理（例如数据集时间范围外），
    直接抛错让问题在 prepare 阶段暴露，而不是训练/评估时出现诡异的空结果。
    """
    start, end = pd.Timestamp(cfg.history_end), pd.Timestamp(cfg.test_start)
    if start >= end:
        raise ValueError('history_end 必须早于 test_start')
    # 跨边界订单整体放到结束时间所在窗口，避免同一订单泄漏到两个窗口。
    # times：每一行所属订单的最晚交易时间（组内 max 广播回行级）。
    times = clean.groupby(['user_id', 'invoice_no']).transaction_time.transform('max')
    # 三段互斥且完备：times < start | start <= times < end | times >= end。
    parts = [clean.loc[times < start].copy(), clean.loc[(times >= start) & (times < end)].copy(),
             clean.loc[times >= end].copy()]
    if any(p.empty for p in parts):
        raise ValueError('时间切分出现空窗口，请调整 history_end/test_start')
    return parts


def sequences(history):
    """把 history 期交易压缩成"订单级商品序列"，作为 Item2Vec 的训练语料。

    参数：
        history : split_data 返回的第一个窗口（建模期交易表）。

    返回：
        list[dict]，每个元素对应一张订单：
        {'user_id': str, 'invoice_no': str, 'items': [商品ID...], 'length': int}。

    概念——为什么订单能当作"句子"：
        Item2Vec 借用 NLP 的 Skip-gram 模型：word2vec 假设"一句话里挨着的词
        语义相关"；对应到电商，"一张订单里一起被购买的商品业务相关"
        （同一批发商同一次采购的商品往往属于同一经营场景）。于是一张订单 =
        一个句子，一个商品 = 一个词。B2B 批发订单平均包含多种商品，共现信号
        比 C 端零售购物车更强，特别适合这种建模。

    实现细节：
        - groupby(sort=True) 按 (user_id, invoice_no) 分组且组顺序确定；
        - 组内再按 (transaction_time, source_row_no) 稳定排序，得到订单内商品的
          录入顺序（word2vec 的窗口机制依赖顺序，顺序不确定则训练不可复现）；
        - dict.fromkeys(...) 是"保序去重"惯用法：dict 保持插入顺序，键天然去重。
          同一订单里同一商品买了多行只保留第一次出现的位置——重复行不提供新的
          共现信息，保留反而会让该商品在序列中自我共现、扭曲向量。
          （不用 set 是因为 set 无序，会破坏序列顺序的确定性。）
        - astype(str)：商品 ID 统一转字符串，gensim 的词表键要求可哈希的字符串。
    """
    result = []
    for (uid, invoice), g in history.groupby(['user_id', 'invoice_no'], sort=True):
        ordered = g.sort_values(['transaction_time', 'source_row_no'], kind='stable')
        items = list(dict.fromkeys(ordered.item_id.astype(str)))
        result.append({'user_id': str(uid), 'invoice_no': str(invoice), 'items': items, 'length': len(items)})
    return result
