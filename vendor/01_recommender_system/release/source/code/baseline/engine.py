"""冻结历史的 Item2Vec、余弦 ItemCF、四路召回与精排特征（推荐引擎核心）。

在流水线中的位置（第 2 阶段 train 的核心实现，也是 web 在线服务的推理内核）：
    输入：history 期交易表（split_data 的第一个窗口）、商品目录 catalog、
          Config、已训练好的 gensim Word2Vec 模型（train_item2vec 的产物）。
    处理：把所有"商品表示"一次性算好并冻结——商品 ID 表、ItemCF 相似度稀疏矩阵、
          Item2Vec 归一化向量矩阵、用户向量、热度/买家数/均价统计；
          然后提供四路召回打分（candidates）、13 维精排特征（features）、
          端到端推荐（recommend）、相似商品检索（similar）、持久化（save/load）。
    输出：召回候选列表、特征 DataFrame、推荐结果、models/baseline 下的
          engine.pkl / vector_index.npz / user_vectors.npz / itemcf.npz。

被谁调用 / 调用谁：
    - pipeline.train() 构造 BaselineEngine 并 save；pipeline.evaluate() 与 web.py
      通过 BaselineEngine.load 恢复后调用 candidates/features/recommend/similar；
    - tests/test_baseline.py 直接构造引擎测试融合、去重、指标与 API；
    - 本模块依赖 .config 的 ROUTES（四路名称与顺序）。

关键概念（详见各函数注释）：
    - Item2Vec / Skip-gram / 负采样：把"一张订单"当"一句话"学商品向量；
    - 余弦相似度：向量归一化后点积即余弦，衡量方向接近程度、与长度无关；
    - ItemCF：物品协同过滤，"买了 A 的人也买 B" 的共现关联；
    - 多路召回 + 融合：先用便宜的算法从全量商品中筛出少量候选，再交给精排模型；
    - log1p：对计数/价格类长尾特征取 log(1+x)，压缩量纲差异；
    - pickle：把 Python 对象整体序列化到磁盘，加载即恢复（含风险，见 save 注释）。
"""
import hashlib
import pickle
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import normalize

from .config import ROUTES

# 精排模型的 13 维特征名，顺序即特征矩阵的列顺序（训练与推理必须完全一致）。
# 前三维是用户侧统计，中间六维是商品侧统计与相似度分，最后四维是"该候选由哪几路
# 召回命中"的 0/1 指示特征（one-hot 风格），用列表解包 *['from_' + r ...] 生成。
# 为什么把"来源路"当特征：不同召回路的先验可靠性不同（例如热门路命中率高但无个性），
# 让精排模型自己学会"哪一路带来的候选更值得排前面"。
FEATURES = ['user_log_orders', 'user_log_items', 'item_log_buyers', 'item_log_orders',
            'item_log_price', 'category_affinity', 'user_cosine', 'itemcf_score',
            'item2vec_score', *['from_' + r for r in ROUTES]]


def stable_hash(value):
    """把字符串映射成一个确定的 64 位整数（SHA-256 摘要的前 8 字节，小端解释）。

    参数：value —— 任意字符串（这里是商品 ID）。
    返回：int，同一输入在任何机器、任何 Python 版本上结果都相同。

    为什么需要它：gensim Word2Vec 内部用 hashfxn 把词映射到"负采样噪声分布表"的
    槽位。Python 内置 hash() 对字符串加了随机盐（PYTHONHASHSEED），每次进程启动
    结果都不同，导致负采样抽到的负例序列不同、训练结果不可复现。换成基于 SHA-256
    的稳定哈希后，固定 seed 即可得到逐位一致的训练结果——这是本课程"可复现性"
    要求的一环。取前 8 字节（64 位）在约 3600 词的词表下碰撞概率可忽略。
    """
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'little')


def top_indices(scores, ids, limit, positive=False):
    """从打分向量里取出分数最高的 limit 个位置下标（带确定性 tie-break）。

    参数：
        scores   : 一维 numpy 数组，每个商品位置一个分数；
        ids      : 与 scores 等长的商品 ID 数组（用于 tie-break 与返回下标对应关系）；
        limit    : 最多返回多少个；
        positive : True 时只保留分数 > 0 的位置（0 分通常意味着"无信号"，
                   例如向量不在词表内或没有任何近邻支持，不应进入召回）。

    返回：一维 numpy 数组，是被选中位置的【下标】（不是分数、不是 ID），
          按分数降序排列；分数相同的按商品 ID 字典序升序——保证结果确定，
          两次运行/两台机器给出的顺序完全一致。

    实现要点：
        - np.isfinite(scores) 先剔除 NaN 与 ±inf。融合前把已购商品置为 -inf
          （见 candidates），这里正是靠 isfinite 把它们排除掉；
        - np.lexsort((次键, 主键))：lexsort 以【最后一个】参数为主排序键，
          所以这里主键是 -scores（取负号 = 降序），次键是 ids（升序）。
          lexsort 是稳定排序，主键相等时才看次键，正是我们要的 tie-break 语义；
        - eligible[order[:limit]]：先取前 limit 名，再映射回原始下标。
    """
    eligible = np.flatnonzero(np.isfinite(scores) & ((scores > 0) if positive else True))
    order = np.lexsort((ids[eligible], -scores[eligible]))
    return eligible[order[:limit]]


def fuse(routes, quotas, final_k):
    """按路截取名额，去重后轮流取数；同商品保留所有命中来源。

    这是"多路召回融合"的实现。概念说明——为什么要多路召回再融合：
        单一算法各有盲区：热门召回覆盖广但完全没有个性化；ItemCF 依赖共购结构，
        对冷门商品无能为力；Item2Vec 依赖向量质量，词表外的新商品直接没有向量；
        用户向量路则容易被历史兴趣主导。把四路各自的 Top-N 汇到一个池子里，
        取长补短，再用一个精排模型统一排序，是工业界标准的"召回-排序"两段式架构。
        召回阶段追求"别漏"（高召回率、便宜），排序阶段追求"排准"（复杂特征、贵）。

    参数：
        routes  : dict，路名 -> 该路按分数降序排好的商品 ID 列表；
        quotas  : dict，路名 -> 该路最多贡献多少个商品（popular 30 / 其余各 40）；
        final_k : 融合后总候选上限（默认配置 150）。

    返回：list[dict]，每项 {'item_id': str, 'sources': [命中的路名, ...]}，
          长度 <= final_k。sources 记录了该商品被哪几路同时召回——多路共同命中
          本身就是强信号，会作为 from_* 特征喂给精排模型。

    为什么用"轮转（round-robin）"而不是简单拼接：
        直接 concat 会让排在前面的路霸占名额（popular 的 30 个全在最前），
        后面的路即使有高分商品也可能被 final_k 截断。轮转按"深度"取数：
        第 0 轮取每路的第 1 名，第 1 轮取每路的第 2 名……这样各路头部结果
        都能公平进入候选池，池子的多样性显著提升。
    为什么去重时保留全部来源而不是只留第一路：
        同一商品被多路召回说明它在多个视角下都相关，丢掉这个信息很可惜；
        保留 sources 让精排能利用"几路共识"这一特征。
    """
    # 每路先保序去重（dict.fromkeys 惯用法，见 data.sequences 注释），再截断到配额。
    # routes.get(r, []) 容忍某路缺失（例如测试里传空列表）。
    selected = {r: list(dict.fromkeys(routes.get(r, [])))[:quotas[r]] for r in ROUTES}
    # 反向索引：商品 -> 命中它的所有路名。setdefault 首次遇到时建空列表再 append。
    sources = {}
    for r, values in selected.items():
        for item in values:
            sources.setdefault(item, []).append(r)
    # 轮转取数：depth 从 0 到"最长那一路的长度-1"，每轮遍历四路各取一个。
    # max(..., default=0) 处理四路全空的边界情况（max() 对空序列会抛异常）。
    output, seen = [], set()
    for depth in range(max((len(x) for x in selected.values()), default=0)):
        for r in ROUTES:
            if depth < len(selected[r]):        # 该路在这个深度还有商品
                item = selected[r][depth]
                if item not in seen:            # 已入选的不重复加入
                    output.append({'item_id': item, 'sources': sources[item]})
                    seen.add(item)
                    if len(output) >= final_k:  # 达到候选池上限立即返回
                        return output
    return output


class BaselineEngine:
    """推荐引擎：把 history 期的全部商品表示冻结下来，供召回、特征与在线服务使用。

    构造时一次性完成所有重计算（ID 表、用户历史、ItemCF 矩阵、向量矩阵、用户向量、
    热度统计），之后每次推荐只做轻量的矩阵索引与点积——这是"离线建索引、在线查索引"
    的标准工程模式，web 服务加载后可对任意用户毫秒级出结果。

    "冻结"的含义与重要性（防泄漏）：
        引擎里的每一个统计量（热度、买家数、均价、向量、近邻）都只来自 history 期
        数据。评估 test 期表现时，引擎完全不知道 test 期发生了什么，因此指标反映的
        是"用过去预测未来"的真实能力。如果热度用了全量数据算，test 期畅销品会
        自动排在前面，指标虚高——这就是未来数据泄漏。
        tests/test_future_changes_do_not_change_history_features 专门验证：
        改动未来窗口的数据，引擎产出的候选与特征必须逐值不变。

    主要属性：
        ids / pos         : 商品 ID 数组及其 ID->下标 映射（全局商品坐标系）；
        users / user_pos  : 用户 ID 列表及其映射；
        catalog           : item_id -> {name, description, category} 展示信息字典；
        histories         : user -> 按时间排序、保序去重的购买商品列表；
        categories        : user -> {类目: 占比}，即用户类目偏好分布；
        orders            : user -> 历史订单数；
        buyers/popularity/prices : 商品侧统计（购买人数、订单热度、平均单价）；
        cf_neighbors      : ItemCF 稀疏相似度矩阵（每行只保留 Top-neighbors）；
        vectors           : 归一化后的 Item2Vec 向量矩阵 (n_items, vector_size)；
        in_vocab          : 布尔数组，该商品是否有有效（非零）向量；
        vector_neighbors  : 向量近邻稀疏矩阵（供 item2vec 召回路）；
        user_vectors      : user -> 归一化用户兴趣向量（最近购买商品向量的均值）。
    """

    def __init__(self, history, catalog, cfg, w2v):
        """构建并冻结引擎的全部索引。

        参数：
            history : history 期清洗后交易表（含 user_id / item_id / invoice_no /
                      transaction_time / UnitPrice / category / source_row_no）；
            catalog : 商品目录表（item_id / description / name / category）；
            cfg     : Config 对象，提供 vector_size / neighbors / recent_items /
                      quotas / k / final_k 等参数；
            w2v     : 已训练好的 gensim Word2Vec 模型，取其 wv（KeyedVectors）
                      里的商品向量。

        设计说明：所有重活都在这里干完，之后的方法都是查表；构造完成后引擎可被
        pickle 整体保存，服务重启直接 load，无需重新训练（节省数分钟训练时间）。
        """
        self.cfg = cfg
        # 全局商品坐标系：把 history 期出现过的商品 ID 排序后编号 0..n-1。
        # sorted() 保证编号顺序确定（不同机器/不同运行完全一致），这是所有矩阵、
        # 向量、npz 文件能互相校验对齐的前提。dtype=str 统一类型，避免 pandas
        # 推断出的混合类型导致 map 失败。
        self.ids = np.array(sorted(history.item_id.unique()), dtype=str)
        self.pos = {x: i for i, x in enumerate(self.ids)}      # ID -> 下标 反查表
        self.users = sorted(history.user_id.unique())
        self.user_pos = {u: i for i, u in enumerate(self.users)}
        # catalog 转成嵌套字典：item_id -> {列名: 值}，展示时 O(1) 查名称/类目。
        self.catalog = catalog.set_index('item_id').to_dict('index')
        self.histories, self.categories, self.orders = {}, {}, {}
        # 每个商品的类目（查不到 catalog 时用 '未知' 兜底）。numpy 数组按位置索引，
        # 与 self.ids 对齐，后面构造特征时用 self.category[i] 直接取。
        self.category = np.array([self.catalog.get(i, {}).get('category') or '未知' for i in self.ids])
        for u, g in history.groupby('user_id', sort=True):
            # 组内按 (时间, 原始行号) 稳定排序：得到该用户购买行为的真实时间顺序。
            ordered = g.sort_values(['transaction_time', 'source_row_no'], kind='stable')
            # 保留最近一次出现的位置，而非把无序 set 当作最近购买。
            # 做法解释：reversed 先倒序 → dict.fromkeys 保序去重（此时留下的是
            # "最后一次出现"的位置，但整体是倒着的）→ [::-1] 再翻转回时间正序。
            # 结果：同一商品重复购买只留最近那次的位置，列表按时间升序。
            # 为什么不用 set：set 无序，取"最近 N 个"会变成随机 N 个，破坏
            # 近因建模与结果可复现性。
            self.histories[u] = list(dict.fromkeys(reversed(ordered.item_id.tolist())))[::-1]
            # 类目偏好：统计该用户历史交易中各类目的出现次数占比（Counter 计数，
            # 除以总行数得到频率分布）。这是"用户-类目亲和度"特征的数据来源。
            counts = Counter(ordered.category)
            self.categories[u] = {c: n / len(ordered) for c, n in counts.items()}
            self.orders[u] = int(g.invoice_no.nunique())       # 订单数 = 不同发票号个数
        # ---- 用户-商品交互矩阵（ItemCF 与买家数统计的基础）----
        # pairs：去重后的 (用户, 商品) 对——即"谁买过什么"，重复购买只算一次。
        pairs = history[['user_id', 'item_id']].drop_duplicates()
        # csr_matrix((值, (行下标, 列下标)), shape=...)：SciPy 的 COO 式稀疏构造，
        # 直接按 (行, 列) 坐标放入数值，这里全部填 1.0（二值矩阵：买过=1）。
        # 为什么用稀疏矩阵：4338 用户 × 3665 商品 ≈ 1600 万个格子，实际交互只有
        # 几万条，稠密存储浪费 99%+ 内存；CSR（压缩稀疏行）格式还特别快于"按行切片"。
        matrix = csr_matrix((np.ones(len(pairs), dtype=np.float32),
                             (pairs.user_id.map(self.user_pos), pairs.item_id.map(self.pos))),
                            shape=(len(self.users), len(self.ids)))
        # buyers：每个商品被多少个【不同用户】买过（对列求和；因为矩阵是二值且
        # pairs 已去重，列和恰好等于购买人数）。asarray(...).ravel() 把
        # (1, n) 的矩阵结果压成一维数组。
        self.buyers = np.asarray(matrix.sum(axis=0)).ravel()
        # popularity：商品热度 = 包含该商品的【订单数】（invoice_no 去重计数），
        # 比行数更能反映"多少次独立采购决策"。reindex(self.ids) 把 groupby 结果
        # 对齐到全局商品坐标系（缺失的商品补 NaN），to_numpy(float) 转成数组。
        # 注意：这里只用 history 期数据统计热度——防泄漏的关键之一。
        self.popularity = history.groupby('item_id').invoice_no.nunique().reindex(self.ids).to_numpy(float)
        # prices：商品历史平均单价（精排特征 item_log_price 的来源）。
        # fillna(0)：理论上 history 里的商品都有价格，防御性处理。
        self.prices = history.groupby('item_id').UnitPrice.mean().reindex(self.ids).fillna(0).to_numpy(float)
        # ItemCF: 二值用户-商品矩阵的列余弦相似度。
        # 概念——余弦相似度：两个向量的夹角余弦 = (a·b)/(|a||b|)，衡量"方向"是否
        # 一致而与"长度"无关。当向量都是单位长度（模=1）时，分母为 1，
        # 点积 a·b 就等于余弦相似度——这就是"归一化后点积=余弦"的由来，
        # 于是可以用一次矩阵乘法批量算出所有商品对的相似度。
        # 概念——ItemCF（物品协同过滤）：商品 A 的列向量记录"哪些用户买过 A"，
        # A、B 两列的余弦越大，说明买 A 的人与买 B 的人重合度越高，
        # 即"买了 A 的人也常买 B"。它完全由共购结构决定，不看商品内容。
        # normalize(matrix, axis=0)：按列（axis=0）把每列缩放成单位向量。
        columns = normalize(matrix, axis=0)
        # columns.T @ columns：(n_items × n_users) @ (n_users × n_items)
        # = n_items × n_items 的稠密相似度矩阵，cf[i][j] = 商品 i 与 j 的余弦。
        # 稀疏矩阵乘稀疏矩阵在 sklearn/scipy 下仍是稀疏的，.toarray() 转稠密以便
        # 后续按行取 Top-N（3665² ≈ 1340 万个 float32，约 54 MB，可接受）。
        cf = (columns.T @ columns).toarray()
        # 对角线是"自己与自己的相似度"恒为 1，必须清零，否则推荐时商品会把自己
        # 排到第一名（我们从不推荐用户已买/查询的商品本身）。
        np.fill_diagonal(cf, 0)
        self.cf_neighbors = self._sparsify(cf, cfg.neighbors)
        # ---- Item2Vec 向量矩阵 ----
        # 先分配全零矩阵（n_items × vector_size），再把 w2v 词表里存在的商品向量
        # 填进对应行。词表外的商品（历史期只出现 1 次、被 min_count=2 过滤掉的
        # 长尾商品）保持全零——零向量与任何向量的点积都是 0，天然"无信号"。
        self.vectors = np.zeros((len(self.ids), cfg.vector_size), dtype=np.float32)
        for item, p in self.pos.items():
            if item in w2v.wv:                    # wv 是 gensim 的 KeyedVectors 词表
                self.vectors[p] = w2v.wv[item]
        # 整体 L2 归一化：每一行缩放成单位向量。之后 self.vectors @ x 就是余弦相似度。
        # 零向量归一化后仍是零向量（sklearn 对零行不做除法，避免除零）。
        self.vectors = normalize(self.vectors)
        # in_vocab：模长 > 0 即"有真实向量"。用于：过滤掉无向量商品的近邻结果、
        # 判断 similar() 的输入是否可查、构造用户向量时跳过无向量商品。
        self.in_vocab = np.linalg.norm(self.vectors, axis=1) > 0
        # 向量相似度矩阵 = 归一化向量的点积 = 余弦相似度（原理同上）。
        sim = self.vectors @ self.vectors.T
        np.fill_diagonal(sim, 0)                  # 同样清零自相似
        self.vector_neighbors = self._sparsify(sim, cfg.neighbors)
        # ---- 用户兴趣向量 ----
        # 做法：取用户最近 recent_items(20) 个已购商品中【有向量】的那些，
        # 求向量均值再归一化。这是最朴素也最常用的"用户表示"方法：
        # 用户 = 他最近买过的东西的平均语义位置。均值会自然放大"反复出现的兴趣方向"，
        # 稀释偶发购买；再归一化是为了后面 self.vectors @ uv 直接得到余弦。
        self.user_vectors = {}
        for u, items in self.histories.items():
            indexes = [self.pos[x] for x in items[-cfg.recent_items:] if self.in_vocab[self.pos[x]]]
            v = self.vectors[indexes].mean(axis=0) if indexes else np.zeros(cfg.vector_size)
            # 除以 max(模长, 1e-12)：手动归一化并防止零向量除零
            # （冷用户/全无向量时 v 是零向量，结果仍是零向量，表示"无兴趣信号"）。
            self.user_vectors[u] = v / max(np.linalg.norm(v), 1e-12)

    def _sparsify(self, dense, neighbors):
        """把稠密相似度矩阵裁剪成"每行只保留 Top-neighbors 正分"的稀疏矩阵。

        参数：
            dense     : (n_items, n_items) 稠密相似度矩阵（对角线已清零）；
            neighbors : 每行保留的近邻个数。

        返回：scipy csr_matrix，形状不变，但每行最多 neighbors 个非零值。

        为什么稀疏化：
            1. 内存——1340 万个 float 降到约 3665×60 ≈ 22 万个；
            2. 质量——相似度 0.01 这种噪声关系留着只会污染召回，
               只保留最强的 60 个近邻，召回结果更聚焦；
            3. 速度——CSR 格式按行求和/取最大值都是稀疏运算，召回时极快。
        positive=True 只保留正分：余弦可能为负（方向相反），负相关不是"相似"，
        不应作为召回依据。
        """
        row, col, val = [], [], []
        for i in range(len(self.ids)):
            chosen = top_indices(dense[i], self.ids, neighbors, positive=True)
            # 把第 i 行选中的近邻以 (行下标, 列下标, 值) 三元组累加进 COO 列表。
            row.extend([i] * len(chosen)); col.extend(chosen); val.extend(dense[i, chosen])
        return csr_matrix((val, (row, col)), shape=dense.shape, dtype=np.float32)

    def scores(self, user_id):
        """为该用户计算四路召回的原始打分向量（尚未截断、未去重、未过滤已购）。

        参数：user_id —— 用户 ID（字符串）。
        返回：dict，路名 -> 长度 n_items 的 float 数组，下标与 self.ids 对齐：
            - 'popular' : 商品历史订单热度（与用户无关，所有用户相同）；
            - 'itemcf'  : ItemCF 分数——用户最近购买商品的近邻相似度【求和】。
                          求和的含义：候选商品被越多历史商品关联、且关联越强，
                          分数越高（多证据累加）；
            - 'item2vec': 向量近邻分数——取最近商品近邻相似度的【最大值】。
                          取 max 而非 sum 是为了只反映"最强的那条语义关联"，
                          避免长历史用户因证据条数多而分数整体膨胀；
            - 'uservec' : 用户兴趣向量与全部商品向量的点积，即余弦相似度
                          （两边都已归一化）。这是唯一"整体兴趣"视角的召回路。

        实现要点：
            - self.cf_neighbors[indexes]：稀疏矩阵按行列表切片，得到
              (最近商品数 × n_items) 的子矩阵；.sum(axis=0) 后 asarray().ravel()
              压成一维（matrix 类型求和会保持二维，必须 ravel）；
            - .max(axis=0).toarray().ravel()：稀疏矩阵按列取最大值，返回稀疏结果，
              toarray 转稠密一维；
            - 用户没有任何历史（冷启动）时，indexes 为空，cf/vec 直接给全零向量，
              此时只有 popular 路与 uservec（零向量 → 全零）参与，实际退化为热门召回。
            - popularity.copy()：返回副本，防止调用方就地修改污染引擎内部状态
              （evaluate 里确实会对分数向量做 values[seen] = -inf 的写操作）。
        """
        history = self.histories.get(user_id, [])
        indexes = [self.pos[i] for i in history[-self.cfg.recent_items:]]
        cf = np.asarray(self.cf_neighbors[indexes].sum(axis=0)).ravel() if indexes else np.zeros(len(self.ids))
        vec = self.vector_neighbors[indexes].max(axis=0).toarray().ravel() if indexes else np.zeros(len(self.ids))
        uv = self.user_vectors.get(user_id, np.zeros(self.cfg.vector_size))
        return {'popular': self.popularity.copy(), 'itemcf': cf, 'item2vec': vec, 'uservec': self.vectors @ uv}

    def candidates(self, user_id, quotas=None, final_k=None):
        """多路召回 + 融合，产出送入精排的候选商品池。

        参数：
            user_id : 用户 ID；
            quotas  : 可选，覆盖默认配额（evaluate 里做"热门配额翻倍"消融实验时传入）；
            final_k : 可选，覆盖默认候选池上限。

        返回（三元组）：
            candidates : fuse() 的输出，list[{'item_id', 'sources'}]，长度 <= final_k；
            scores     : scores() 的四路原始打分向量（精排特征要用其中三路当特征值）；
            routes     : dict，路名 -> 该路截断后的商品 ID 列表（融合前的原始各路结果，
                         evaluate 里单独评估 popular / itemcf 策略时直接取用）。

        为什么必须排除用户已购商品：
            推荐"新品发现"场景下，再推用户刚买过的东西是浪费曝光（B2B 补货是例外，
            那是 replenishment.py 单独一路的设计动机）。实现上把已购位置的分数置为
            -inf，top_indices 里的 np.isfinite 过滤会自动把它们剔除——用 -inf 而不是
            删元素，是为了保持向量长度与 self.ids 对齐，索引关系不被破坏。

        为什么每路取 max(配额, cfg.k)：
            配额是"给融合池的名额"，而 routes 结果还要单独作为 popular / itemcf
            策略的 Top-K 输出参与评估。若某路配额小于 k（本配置不会，但要防御），
            单独评估时就凑不齐 K 个，所以至少取 k 个。

        positive=r != 'popular'：
            itemcf / item2vec / uservec 三路里，0 分表示"完全无信号"（没有共购关联、
            向量不在词表、用户无历史），把这些商品塞进候选纯属占位，故只取正分；
            popular 路的热度理论上恒 >= 0，且热门是必须保留的对照基线，不做正分过滤
            （否则冷启动用户会一路候选都没有）。
        """
        quotas = self.cfg.quotas() if quotas is None else quotas
        scores = self.scores(user_id)
        seen = [self.pos[x] for x in self.histories.get(user_id, [])]
        routes = {}
        for r in ROUTES:
            values = scores[r].copy()          # 副本：就地写 -inf 不影响 scores 返回值
            values[seen] = -np.inf             # 屏蔽历史已购商品
            idx = top_indices(values, self.ids, max(quotas[r], self.cfg.k), positive=r != 'popular')
            routes[r] = self.ids[idx].tolist()
        return fuse(routes, quotas, final_k or self.cfg.final_k), scores, routes

    def features(self, uid, candidates, scores):
        """为"用户 × 候选商品"的每一对构造 13 维精排特征矩阵。

        参数：
            uid        : 用户 ID；
            candidates : candidates() 输出的候选列表（提供 item_id 与 sources）；
            scores     : scores() 输出的四路打分向量（其中三路直接当特征值）。

        返回：pandas DataFrame，行数 = 候选数，列名严格等于 FEATURES（顺序一致），
              无 NaN（最后 fillna(0) 兜底）。精排模型（LR / LightGBM）的 predict_proba
              直接吃这个矩阵。

        13 维特征的业务含义：
            user_log_orders   用户历史订单数（活跃度）——对数压缩后；
            user_log_items    用户历史购买过的不同商品数（品类广度）；
            item_log_buyers   该商品被多少个不同用户买过（受众广度）；
            item_log_orders   该商品出现在多少张订单里（热度）；
            item_log_price    该商品历史平均单价（价格带）；
            category_affinity 用户在该商品所属类目上的历史购买占比（0~1）；
            user_cosine       用户兴趣向量与该商品向量的余弦相似度（整体兴趣匹配）；
            itemcf_score      该商品的 ItemCF 召回分（共购结构关联强度）；
            item2vec_score    该商品的向量近邻召回分（语义关联强度）；
            from_popular / from_itemcf / from_item2vec / from_uservec
                              四个 0/1 指示特征：该候选是否由对应召回路命中。

        概念——log1p（为什么对计数特征取对数）：
            订单数、买家数这类计数特征分布极度右偏（长尾）：绝大多数商品被 1~2 人
            买过，个别爆款被几百人买过。直接喂给线性模型，爆款的数值会主导整个
            特征尺度，模型学到的是"越热门越好"的粗暴规律。取对数把乘性差异变成
            加性差异（1000 vs 100 的差距 ≈ 100 vs 10 的差距），分布接近正态，
            线性模型和梯度下降都更好处理。
            用 log1p = log(1+x) 而不是 log(x)：x=0 时 log(0) = -inf 会炸掉训练；
            +1 平移保证 log1p(0)=0 有限且单调，同时 x 较大时 log1p(x) ≈ log(x)，
            几乎不损失信息。价格特征虽非计数，但同样右偏，一并取对数。

        概念——为什么用户级特征要用 np.full 广播：
            user_log_orders / user_log_items 对一个用户是常数，但特征矩阵是
            "每候选一行"，所以要把标量重复成与候选等长的列（np.full）。
            这不是冗余设计：LR 需要每行特征维度一致，用户级常数项相当于该用户的
            偏置修正，在跨用户混合训练时是有信息量的。

        实现细节：np.column_stack 按列拼接成一维等长数组的二维矩阵，
            列顺序必须与 FEATURES 完全一致（这是训练/推理特征对齐的硬约束）；
            *[[...] for r in ROUTES] 解包出四个 0/1 列，顺序同样由 ROUTES 决定。
        """
        indexes = np.array([self.pos[c['item_id']] for c in candidates], dtype=int)
        affinity = self.categories.get(uid, {})     # 用户类目偏好分布（冷用户为空 dict）
        rows = np.column_stack([
            np.full(len(indexes), np.log1p(self.orders.get(uid, 0))),
            np.full(len(indexes), np.log1p(len(self.histories.get(uid, [])))),
            np.log1p(self.buyers[indexes]), np.log1p(self.popularity[indexes]), np.log1p(self.prices[indexes]),
            [affinity.get(self.category[i], 0) for i in indexes],
            scores['uservec'][indexes], scores['itemcf'][indexes], scores['item2vec'][indexes],
            *[[int(r in c['sources']) for c in candidates] for r in ROUTES]])
        return pd.DataFrame(rows, columns=FEATURES).fillna(0)

    def similar(self, item_id, k=5):
        """查询与指定商品最相似的 k 个商品（Item2Vec 向量近邻检索）。

        参数：
            item_id : 查询商品 ID；
            k       : 返回个数。

        返回：list[dict]，每项是 enrich() 格式的商品信息（含 score = 余弦相似度），
              按相似度降序、ID 字典序 tie-break。

        异常：商品不在 pos（history 期未出现）或不在词表（向量全零）时抛 KeyError，
              web 层捕获后返回 404——语义是"这个商品没有可比较的向量"，
              而不是服务器错误，用 404 比 500 更准确。

        实现要点：
            self.vectors @ self.vectors[pos]：矩阵乘单个向量 = 该商品与全部商品的
            余弦相似度（因为向量都已 L2 归一化，点积即余弦）。
            随后把"无向量的商品"和"自己"置为 -inf 排除：
            - ~self.in_vocab：零向量点积为 0，虽然不会排到前面，但显式排除更保险；
            - 自己：查询"与 A 最相似的商品"不应该返回 A 本身（自相似恒为 1，
              会永远霸占第一名）。测试 test_vector_index_roundtrip_and_self_filter
              专门验证这一点。
        """
        if item_id not in self.pos or not self.in_vocab[self.pos[item_id]]:
            raise KeyError('商品不在 Item2Vec 词表中')
        values = self.vectors @ self.vectors[self.pos[item_id]]
        values[~self.in_vocab] = -np.inf
        values[self.pos[item_id]] = -np.inf
        return [self.enrich(self.ids[i], float(values[i])) for i in top_indices(values, self.ids, k)]

    def enrich(self, item, score, sources=None):
        """把"商品 ID + 分数"包装成带展示信息的字典（API 返回的统一商品格式）。

        参数：
            item    : 商品 ID；
            score   : 该商品的分数（相似度/热度/精排概率，视调用场景而定）；
            sources : 可选，命中的召回路名列表；None 时给空列表。

        返回：{'item_id', 'name', 'description', 'category', 'score', 'sources'}。
              name 按 中文名 → 英文描述 → '未知商品' 三级兜底（与 clean_data 的
              展示口径一致）；category 缺失给 '未知'；score 强制转 float
              （numpy 标量不能被 json 直接序列化，这是 API 层常见的坑）。
        """
        p = self.catalog.get(item, {})
        return {'item_id': str(item), 'name': p.get('name') or p.get('description') or '未知商品',
                'description': p.get('description') or '',
                'category': p.get('category') or '未知', 'score': float(score), 'sources': sources or []}

    def recommend(self, uid, ranker, k=10):
        """端到端推荐：召回 → 构造特征 → 精排打分 → 取 Top-K。

        参数：
            uid    : 用户 ID；
            ranker : 训练好的精排模型（sklearn Pipeline，须实现 predict_proba）；
            k      : 返回条数（课程评估用 10）。

        返回：list[dict]（enrich 格式，score = 精排模型输出的"会购买"概率），
              按分数降序。候选为空时返回空列表。

        概念——为什么分"召回 + 精排"两段：
            对全量 3665 个商品逐一算 13 维特征再过模型，代价高且大部分商品明显不相关。
            工业界通行做法是漏斗式：召回阶段用便宜的向量/统计方法快速筛出上百个候选
            （追求不漏），精排阶段用带交叉特征的监督模型精细排序（追求排准）。
            本项目的 final_k=150 与 Top-10 输出正是这个漏斗的两级。

        实现要点：
            predict_proba(x)[:, 1]：sklearn 分类器的 predict_proba 返回形状
            (n_samples, n_classes) 的矩阵，列顺序按 classes_（[0, 1]），
            第 1 列即"正类（会购买）"的概率。这个概率被当作排序分数使用——
            注意由于训练时对负样本做了降采样，它的绝对数值不是真实购买概率，
            只有相对大小有意义（报告里明确说明了这一边界）。
        """
        candidates, scores, _ = self.candidates(uid)
        if not candidates:
            return []
        x = self.features(uid, candidates, scores)
        values = ranker.predict_proba(x)[:, 1]
        ids = np.array([c['item_id'] for c in candidates])
        order = top_indices(values, ids, k)       # 分数降序 + ID tie-break，取前 k
        return [self.enrich(ids[i], values[i], candidates[i]['sources']) for i in order]

    def save(self, directory):
        """把引擎持久化到目录：pickle 快照 + 三个 numpy/稀疏矩阵文件。

        参数：directory —— Path，不存在时自动创建（models/baseline）。

        产物：
            engine.pkl        : 引擎对象整体的 pickle 序列化；
            vector_index.npz  : ids / vectors / in_vocab（向量检索索引）；
            user_vectors.npz  : 用户 ID 与用户向量；
            itemcf.npz        : ItemCF 稀疏相似度矩阵。

        概念——pickle 是什么、有什么风险：
            pickle 是 Python 的对象序列化机制：把内存中的对象图（含 numpy 数组、
            dict、类实例）原样写成字节流，pickle.load 时在【当前进程】重建等价对象。
            好处是一行代码保存/恢复整个引擎，无需重新训练（省数分钟）；
            风险有两点：
            1. 安全——反序列化会执行类定义相关的代码，绝不能 load 来源不明的
               pkl 文件（等于运行别人的程序）。本项目只加载自己训练产出的文件；
            2. 兼容——pkl 与类定义、库版本绑定，改了 BaselineEngine 的字段或
               升级 sklearn/numpy 后旧 pkl 可能加载失败或行为不一致。
               因此项目同时把关键矩阵另存为 npz（纯数据格式，跨版本稳定），
               load 时校验 ids 是否一致，并在 manifest.json 记录全部库版本。
        为什么向量矩阵要单独存 npz 而不是只靠 pkl：
            npz 是 numpy 原生的多数组压缩存档，加载快、可被 np.load(allow_pickle=False)
            安全读取（不执行任何代码），也便于 web 端只更新向量索引而不动引擎其余部分。
        """
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'engine.pkl').open('wb') as f:
            pickle.dump(self, f)
        np.savez_compressed(directory / 'vector_index.npz', ids=self.ids, vectors=self.vectors, in_vocab=self.in_vocab)
        np.savez_compressed(directory / 'user_vectors.npz', ids=np.array(self.users),
                            vectors=np.array([self.user_vectors[u] for u in self.users]))
        save_npz(directory / 'itemcf.npz', self.cf_neighbors)

    @staticmethod
    def load(directory):
        """从目录恢复引擎（不重新训练、不重建索引），并做一致性校验。

        参数：directory —— save() 用过的目录（Path）。
        返回：BaselineEngine 实例，可直接用于 candidates / recommend / similar。

        步骤与设计理由：
            1. pickle.load 恢复引擎对象整体（含 cfg、histories、catalog 等）；
            2. 从 vector_index.npz 重新读入向量矩阵，并断言 npz 里的 ids 与 pkl 里
               的 engine.ids 完全相等——若不等说明两个文件来自不同训练批次，
               按位置索引会张冠李戴（把 A 商品的向量当成 B 的），必须硬失败。
               注释原文：恢复已保存的向量检索矩阵，不重新训练或重建索引。
               .copy()：np.load 返回的是内存映射数组，文件句柄关闭后可能失效，
               复制成独立内存更安全；
            3. allow_pickle=False：明确禁止 npz 里夹带 pickle 对象，
               只接受纯数组（安全最佳实践）；
            4. 正式模型目录（MODELS）下额外从 products.sqlite3 重新读商品展示信息：
               服务启动时从持久化商品库读展示字段；测试临时目录使用随模型保存的目录。
               这样商品名/类目可以在不改模型的前提下更新（sqlite3 用 mode=ro
               只读 URI 打开，杜绝服务进程误写数据库）；测试用的 tmp_path 没有
               sqlite3 文件，就沿用 pkl 里冻结的 catalog，逻辑自动分支。
        """
        with (directory / 'engine.pkl').open('rb') as f:
            engine = pickle.load(f)
        # 恢复已保存的向量检索矩阵，不重新训练或重建索引。
        with np.load(directory / 'vector_index.npz', allow_pickle=False) as z:
            if not np.array_equal(z['ids'], engine.ids):
                raise ValueError('索引与引擎商品顺序不一致')
            engine.vectors, engine.in_vocab = z['vectors'].copy(), z['in_vocab'].copy()
        # 服务启动时从持久化商品库读展示字段；测试临时目录使用随模型保存的目录。
        from .config import MODELS, PROCESSED
        if directory.resolve() == MODELS.resolve():
            with sqlite3.connect(f'file:{(PROCESSED / "products.sqlite3").as_posix()}?mode=ro', uri=True) as con:
                engine.catalog = pd.read_sql_query('SELECT * FROM products', con).set_index('item_id').to_dict('index')
        return engine


def train_item2vec(seqs, cfg):
    """用订单商品序列训练 Item2Vec（gensim Word2Vec 的 Skip-gram + 负采样）。

    参数：
        seqs : data.sequences() 的输出，list[{'user_id','invoice_no','items','length'}]；
        cfg  : Config，使用 vector_size / window / min_count / negative / epochs /
               seed 六个字段。

    返回：训练好的 gensim Word2Vec 模型（模型.wv 是商品向量词表）。

    概念——Item2Vec / Skip-gram 在做什么：
        把一张订单当成一个"句子"、一个商品当成一个"词"。Skip-gram 的训练任务是：
        给定中心词，预测它窗口内（前后各 window=5 个位置）的上下文词。
        模型为每个商品学一个 64 维向量，训练目标是让"经常在同一订单中相近位置
        出现的商品"向量方向接近（点积大），"随机凑对的商品"向量方向远离。
        训练完成后，向量距离就近似"业务相关性"：一起被批发采购的商品
        （同属一个经营场景）会聚在一起，于是可以做近邻检索与召回。

    概念——负采样（negative sampling），为什么需要、采几个：
        词表有几千个商品，如果用完整 softmax 预测"下一个商品是哪个"，每次更新都要
        对全部几千个词算归一化分母，代价极高。负采样把它改写成一个廉价的二分类问题：
        对每个真实共现对 (中心商品, 上下文商品) 记为正例，再从词表按词频的 3/4 次方
        分布随机抽 negative=5 个"没在这对里共现"的商品当负例，训练目标变成
        "把正例的点积推高、把这 5 个负例的点积压低"。
        为什么"没共现的商品对"能当负例：语言/购物数据里，随机两个商品同时出现在
        同一窗口的概率极低，随机抽到的几乎必然是真正的负相关对（这就是负采样成立
        的统计前提）。抽 5 个是经验折中：负例太少，向量空间区分度不够（什么都能
        挨近）；太多，训练变慢且会把大量弱相关商品过度推开。按频率的 3/4 次方抽样
        则让热门商品被抽为负例的概率略低于线性——避免冷门商品的向量被过度挤压。

    各参数的作用（对应 cfg 字段）：
        vector_size=64  向量维度；
        window=5        共现窗口（订单内前后 5 个位置算共现）；
        min_count=2     出现少于 2 次的商品不入词表（长尾噪声，且无法学到可靠向量）；
        sg=1            使用 Skip-gram（sg=0 则是 CBOW：用上下文预测中心词，
                        方向相反；Skip-gram 对低频词更友好，适合长尾商品多的场景）；
        negative=5      负采样个数（见上）；
        epochs=10       全语料遍历 10 轮；
        seed=cfg.seed   随机种子，配合下面的稳定哈希保证完全可复现；
        workers=1       单线程训练。多线程下 gensim 的负例抽取顺序依赖线程调度，
                        结果会有微小抖动；这里数据量不大，牺牲一点速度换确定性；
        sorted_vocab=1  词表按词排序建索引（而非按出现顺序），进一步消除
                        语料顺序对内部编号的影响；
        hashfxn=stable_hash  用 SHA-256 稳定哈希替代 Python 内置 hash（见该函数注释）。

    过滤 len(items) > 1：只含一个商品的订单没有任何共现关系，对 Skip-gram
        提供不了训练信号（中心词没有上下文），留着只会拖慢训练。
    """
    sentences = [s['items'] for s in seqs if len(s['items']) > 1]
    return Word2Vec(sentences=sentences, vector_size=cfg.vector_size, window=cfg.window,
                    min_count=cfg.min_count, sg=1, negative=cfg.negative, epochs=cfg.epochs,
                    seed=cfg.seed, workers=1, sorted_vocab=1, hashfxn=stable_hash)
