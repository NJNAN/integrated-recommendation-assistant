"""全局配置与路径常量 —— 整条推荐流水线的"唯一事实来源"（single source of truth）。

在流水线中的位置：
    prepare(数据准备) -> train(训练) -> evaluate(评估) -> web(在线演示) 的每一个阶段，
    都首先依赖本模块。它本身不含业务逻辑，只定义两类东西：
    1. 目录常量（ROOT/RAW/PROCESSED/MODELS/OUTPUTS）：规定原始数据、清洗后的中间产物、
       训练好的模型、评估报告分别存放在磁盘的哪个位置；
    2. Config 数据类：把整条流水线所有可调的超参数集中在一处（时间切分边界、
       Item2Vec 训练参数、四路召回配额、融合候选上限、评估设置等）。

被谁调用 / 调用谁：
    - 本模块不 import 任何项目内模块（处于依赖链最底层）；
    - pipeline.py 读取 Config 执行三个阶段，并把最终生效的配置序列化写入 config.json；
    - engine.py 读取 ROUTES 和各 *_quota 字段完成四路召回与融合；
    - data.py 读取 history_end / test_start 完成时间切分；
    - web.py 读取 MODELS / OUTPUTS 定位训练产物并加载模型。

输入 / 输出：
    无外部输入。输出是纯 Python 对象（Path 常量与 Config 实例）；
    pipeline 会把 Config.asdict() 持久化为 JSON，保证"训练时用的参数"事后可追溯、
    且 evaluate 阶段能校验加载的模型与当前配置是否一致。
"""
from dataclasses import asdict, dataclass
from pathlib import Path

# ROOT：项目代码根目录（即 code 文件夹）。
# Path(__file__) 是 config.py 自身的路径；.resolve() 把它规范化为绝对路径；
# .parents[1] 表示向上数两级目录：parents[0] 是 baseline/，parents[1] 是 code/。
# 用 __file__ 定位（而不是依赖当前工作目录 os.getcwd()），保证无论从哪个目录启动
# python -m baseline.pipeline，路径都指向同一处，避免"换个目录就跑不起来"。
ROOT = Path(__file__).resolve().parents[1]
# RAW：原始数据目录，存放未经任何修改的 transactions.csv（39.8 万条交易）与
# products.csv（商品主数据）。原始数据只读不写，是所有处理的起点。
RAW = ROOT / 'data' / 'raw'
# PROCESSED：清洗、切分后的中间产物目录（history.csv / rank_train.csv / test.csv /
# catalog.csv / user_sequences.csv / config.json / recall_set.csv 等）。
# 这些文件由 prepare 阶段生成，train / evaluate / web 阶段读取。
PROCESSED = ROOT / 'data' / 'processed' / 'baseline'
# MODELS：训练好的模型文件目录（engine.pkl 引擎快照、item2vec.model 词向量、
# rankers.pkl 精排模型、*.npz 稀疏/稠密矩阵索引）。
MODELS = ROOT / 'models' / 'baseline'
# OUTPUTS：评估结果与报告目录（evaluation.json、baseline_report.md、per_user_metrics.csv、
# run_status.json 等），面向"人看"的产物都放这里。
OUTPUTS = ROOT / 'outputs' / 'baseline'


# @dataclass：Python 标准库装饰器，根据类里声明的字段自动生成 __init__ / __repr__ /
# __eq__ 等样板代码，省去手写构造器。
# frozen=True 表示"冻结"：实例创建后任何属性赋值都会抛出 FrozenInstanceError。
# 这样配置对象在多个模块间传递时不可能被悄悄改动，保证训练与评估使用的参数完全一致
# （evaluate 中 engine.cfg != cfg 的一致性校验正是依赖 __eq__ 逐字段比较）。
@dataclass(frozen=True)
class Config:
    """整条流水线的全部超参数，集中为一个不可变对象。

    设计理由：与其把参数散落在各模块的模块级常量里，不如集中成一个数据类——
    它可以整体序列化（asdict -> config.json）随产物落盘，加载模型时再做一致性校验，
    防止"用一套参数训练、另一套参数评估"这种隐蔽错误；同时所有实验只需 replace()
    出一个新 Config 即可，不改代码。

    字段含义见各字段的行内注释。
    """

    seed: int = 42                    # 全局随机种子。控制评测用户抽样、精排负样本抽样、
                                      # Item2Vec 初始化等一切随机环节；固定种子后每次运行
                                      # 结果完全一致（可复现性是课程工程要求之一）。
    history_end: str = '2011-08-01'   # 历史（建模）期截止时间：早于该时刻的交易用于训练
                                      # 商品向量、ItemCF、热度统计——所有商品表示"冻结"在此。
    test_start: str = '2011-10-01'    # 最终测试期起始时间：评估只看该时刻之后的真实购买。
    # 三个时间窗口的划分：history(< history_end) -> rank 标签期 [history_end, test_start)
    # -> test(>= test_start)。标签期专门给精排模型（LR/LightGBM）产生训练标签。
    # 为什么切三段而不是两段：如果直接用测试期训练精排模型，模型就"提前看过考卷答案"，
    # 离线指标会虚高（数据泄漏）；三段式让"建表示、学排序、最终打分"各用互不重叠的时间段。
    vector_size: int = 64             # Item2Vec 商品向量的维度。维度越高表达能力越强，
                                      # 但更容易过拟合、训练与检索更慢；64 是中等规模
                                      # 词表（约 3600 个商品）下的常用经验值。
    window: int = 5                   # Skip-gram 上下文窗口大小：训练时把同一订单序列中
                                      # 每个商品前后最多 5 个位置的商品视为"共现"正例。
                                      # 窗口越大，学到的是越宽泛的同类关联；越小越强调
                                      # 紧邻的强搭配。
    epochs: int = 10                  # 训练轮数：全部订单序列被模型完整遍历 10 次。
    min_count: int = 2                # 进入词表的最小出现次数：整个历史期只出现 1 次的
                                      # 商品被视为噪声/长尾，不分配向量（也节省内存）。
    negative: int = 5                 # 负采样个数：每个"真实共现"正例，从词表按频率随机
                                      # 抽 5 个未共现商品当负例，训练目标是把正例向量拉近、
                                      # 负例推远（详见 engine.train_item2vec 的注释）。
    neighbors: int = 60               # 相似度矩阵稀疏化时每个商品保留的近邻个数：全量
                                      # 3600x3600 相似度矩阵中绝大多数是小值噪声，只留
                                      # Top-60 既省内存又让召回更聚焦。
    recent_items: int = 20            # 构建"用户向量"与触发 ItemCF/Item2Vec 召回时，只取
                                      # 用户最近购买的 20 个商品。近因原则：越近的购买越能
                                      # 代表当前兴趣，也让计算量与用户历史长度解耦。
    popular_quota: int = 30           # 热门召回路配额：向融合池最多贡献 30 个按历史订单数
                                      # 排序的商品。热门路是无个性化的"保底"，也是评估时
                                      # 所有个性化策略必须战胜的对照基线。
    itemcf_quota: int = 40            # ItemCF（物品协同过滤）召回路配额。
    item2vec_quota: int = 40          # Item2Vec 向量近邻召回路配额。
    uservec_quota: int = 40           # 用户向量召回路（用户兴趣向量与全部商品向量点积）配额。
    final_k: int = 150                # 多路融合去重后送入精排的候选总数上限。精排模型要对
                                      # 每个候选逐一算特征、打分，代价与候选数成正比；
                                      # 150 是"召回覆盖率"与"精排开销"的折中。
    negative_ratio: int = 5           # 精排训练时正负样本比例：每个正例最多配 5 个从候选中
                                      # 抽出的负例。候选中绝大多数商品用户未来并不会买，
                                      # 若全量当负例，正负比会悬殊到 1:100+，模型会倾向全预测
                                      # "不买"；降采样负例让训练信号更均衡。
    eval_users: int = 500             # 评估时抽样的测试用户个数（用固定种子从全部符合
                                      # 条件的用户中随机抽取）。0 = 全部符合评测条件的用户。
                                      # 抽样是为了控制评估耗时（每个用户要跑一次完整召回），
                                      # 500 人的均值已足够稳定地比较各策略。
    k: int = 10                       # Top-K 评估截断：只看每个用户推荐列表的前 10 个是否
                                      # 命中真实购买，对应报告中的 P@10 / R@10 / NDCG@10。

    def quotas(self):
        """把四路召回的配额字段汇总成字典：{'popular': 30, 'itemcf': 40, ...}。

        用 getattr 按 "路名 + '_quota'" 动态读取属性，避免手写四行重复代码，
        新增召回路时只要在 ROUTES 和字段里各加一项即可。
        注意：ROUTES 定义在本类之后（文件末尾），但这不影响运行——Python 是在
        函数体被调用的那一刻才去全局作用域查找 ROUTES，而不是在类定义时。

        返回：dict，键为路名，值为该路的候选配额上限。
        """
        return {r: getattr(self, r + '_quota') for r in ROUTES}

    def asdict(self):
        """把配置对象转成普通 dict（dataclasses.asdict 递归转换）。

        用途：json.dumps 序列化后写入 config.json 与 manifest.json，让每次运行的
        全部参数可追溯；evaluate 阶段再用它重建 Config 与模型内保存的配置比对。
        """
        return asdict(self)


# 四路召回的固定顺序。用元组（tuple）而非列表，表示这是"约定，不应被修改"；
# 融合时的轮转顺序（engine.fuse）、精排特征中 from_* 指示列的顺序都以此为准，
# 保证特征列顺序在任何一次运行中完全一致（模型 pickle 后重新加载也依赖这一点）。
ROUTES = ('popular', 'itemcf', 'item2vec', 'uservec')
