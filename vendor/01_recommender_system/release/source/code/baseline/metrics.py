"""评估指标：Top-K 排序质量（Precision / Recall / NDCG / HitRate）与列表重叠率。

在流水线中的位置（第 3 阶段 evaluate 的计算内核）：
    输入：每个用户的推荐列表（商品 ID 序列）与真实购买集合（test/标签窗口内、
          历史期未购买过的商品），以及对比用的两个列表。
    输出：dict 形式的单用户指标（pipeline.evaluate 汇总成 per_user_metrics.csv 与
          evaluation.csv），以及两个策略列表的重叠率（用于个性化程度分析）。

被谁调用 / 调用谁：
    - pipeline.evaluate() 对 500 个抽样测试用户 × 每个策略逐一调用 topk_metrics
      与 overlap；pipeline.write_report() 把汇总结果渲染进 Markdown 报告；
    - tests/test_baseline.py 用手工可算的小例子验证指标数值正确；
    - 本模块只依赖标准库 math，不依赖项目内任何其他模块（纯函数，最易测试）。

关键概念——推荐系统的离线评估范式：
    把"系统给出的 Top-K 列表"与"用户未来真实购买的商品集合"对比。
    注意这里没有分级相关性（只有"买了/没买"二值），所以 NDCG 的增益项取 0/1，
    公式随之简化（详见 topk_metrics 注释）。
"""
import math


def topk_metrics(recommendations, truth, k):
    """计算单个用户 Top-K 推荐列表的四个评估指标。

    参数：
        recommendations : 系统给出的推荐商品 ID 列表（已按推荐分数降序排列）；
        truth           : 该用户在评估窗口内真实购买的商品集合（可迭代对象，
                          内部转成 set 以便 O(1) 成员判断）；
        k               : 截断位置，只评估列表前 k 个（课程口径 K=10）。

    返回：dict，含四个指标（均为 float，取值 [0,1]）：
        precision_k : 前 k 个推荐中命中真实购买的【比例】——"推荐的准不准"。
                      分母恒为 k（哪怕列表不足 k 个，缺席位置算未命中，
                      这是标准做法，惩罚凑不齐 K 个的策略）；
        recall_k    : 真实购买商品中被前 k 个推荐覆盖的【比例】——"该推的漏没漏"。
                      truth 为空集时定义返回 0.0（调用方 evaluate 只对 truth 非空的
                      用户评估，这里的 0.0 只是防御性兜底，避免除零）；
        ndcg_k      : 归一化折损累计增益——在 Precision 基础上进一步考虑
                      【命中发生在第几位】，越靠前的命中贡献越大（详见下方概念）；
        hit_rate_k  : 前 k 个中只要命中至少一个就是 1.0，否则 0.0——
                      "这个用户有没有被推荐打动"的最宽松口径，对业务而言
                      常用来衡量推荐触达率。

    异常：k < 1 时抛 ValueError（K=0 无意义且会导致除零）。

    概念——NDCG（Normalized Discounted Cumulative Gain）与"位置折扣"：
        直觉：把正确答案排在第 1 位比排在第 10 位好得多——用户视线从上往下衰减，
        列表越靠后的位置被看到的概率越低。NDCG 用数学方式表达这个直觉：
        1. 增益（Gain）：每个位置的命中记 1，未命中记 0（二值相关性场景）；
        2. 折扣（Discount）：第 rank 位（从 0 开始计数）的增益除以 log2(rank + 2)。
           为什么是 rank+2：rank 从 0 起，而 log2(1)=0 会除零，所以平移成
           log2(2)=1、log2(3)≈1.585、log2(4)=2……位置越靠后，折扣分母越大，
           同样一个命中贡献的分数越小——这就是"位置折扣"；
        3. 累计（Cumulative）：把 k 个位置的折损增益求和，得到 DCG；
        4. 归一化（Normalized）：除以理想排序（IDCG）——假设全部命中商品都排在
           最前面时的 DCG，即 min(k, len(truth)) 个位置全为 1 的折损和。
           归一化把分数拉到 [0,1]，不同用户（真实购买数不同）之间才可比：
           买了 30 件商品的用户命中 1 件与只买 2 件的用户命中 1 件，
           Precision 相同但 Recall 天差地别，NDCG 通过各自的理想值校准了这一点。
        本实现中 dcg = Σ hit / log2(rank+2)，ideal = Σ 1 / log2(rank+2)
        （求和上限取 min(k, len(truth))：命中最多不超过真实商品数，也不超过 k），
        ndcg_k = dcg / ideal；ideal 为 0（truth 为空）时返回 0.0。

    实现细节：
        recs = list(dict.fromkeys(recommendations))[:k]
        ——先去重再截断。dict.fromkeys 保序去重（见 data.sequences 注释）：
        若推荐列表意外含重复商品，重复项会虚占 k 个坑位、抬高 Precision，
        先去重保证"k 个坑位对应 k 个不同商品"。tests 中
        test_known_metrics_and_duplicate_predictions 用 ['A','A','X','B'] 验证：
        去重后前 3 位是 A、X、B，命中 A、B 两个，Precision = 2/3 而不是 3/3。
        hits = [int(i in truth) for i in recs]：把"是否命中"变成 0/1 列表，
        int(bool) 是 Python 惯用转换（True→1, False→0），方便后续求和与逐位折扣。
    """
    if k < 1:
        raise ValueError('k 必须大于 0')
    recs = list(dict.fromkeys(recommendations))[:k]
    truth = set(truth)
    hits = [int(i in truth) for i in recs]
    # DCG：对每个位置，命中(1) 除以位置折扣 log2(rank+2) 后累加；未命中贡献 0。
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    # IDCG（理想 DCG）：假设前 min(k, 真实商品数) 个位置全部命中的折损和，
    # 即该用户在该 k 下能达到的理论最高 DCG，用作归一化分母。
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(truth))))
    return {'precision_k': sum(hits) / k, 'recall_k': sum(hits) / len(truth) if truth else 0.0,
            'ndcg_k': dcg / ideal if ideal else 0.0, 'hit_rate_k': float(any(hits))}


def overlap(a, b, k):
    """计算两个推荐列表前 k 项的重叠率（Jaccard 式重合度）。

    参数：
        a, b : 两个推荐商品 ID 列表（均已按各自分数降序）；
        k    : 只比较各自的前 k 项。

    返回：float ∈ [0,1]，= |a 前 k 项集合 ∩ b 前 k 项集合| / k。
          1 表示两个策略给出完全相同的 Top-K，0 表示毫无交集。

    在本项目中的两个用途（见 pipeline.evaluate）：
        1. popular_overlap：每个个性化策略（itemcf / lr / lightgbm）与【同一用户的】
           热门召回 Top-K 的重叠率——衡量"个性化程度"。如果某策略输出与热门几乎
           一样，说明它没学到用户特异性信号，复杂模型的价值存疑；
        2. lr_lightgbm_overlap：两个精排模型之间的重叠率——衡量模型差异度。
           重叠率越高说明两模型排序结论越接近（不等于质量越高，报告里明确了这一点）。

    概念提示：重叠率是"列表相似度"指标，与 topk_metrics 的"列表质量"指标正交——
        质量回答"推得准不准"，重叠回答"和别的策略像不像"，两者要一起看：
        一个策略可能质量与热门相当但重叠极低（找到了不同的正确答案），
        也可能重叠很高纯粹因为热门本身命中率高。评估报告同时呈现两者正是为此。

    注意：本函数假定调用方传入 k >= 1（evaluate 传入 cfg.k=10），未做防御检查；
        若 a 或 b 不足 k 项，缺失位置自然无法重叠，分母仍为 k——与 Precision 的
        口径一致（凑不齐 K 个会被计入不重叠）。
    """
    return len(set(a[:k]) & set(b[:k])) / k
