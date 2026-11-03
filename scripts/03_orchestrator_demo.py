"""阶段3 验收（可重跑）：正常推荐、硬约束空集、上游超时降级、确定性复现。

跑法：`python -B scripts/03_orchestrator_demo.py`
产物：`outputs/stage03_demo.json`（每条用例含 input / expected_status / response / passed）
约定：4 条用例必须全过；任何一条失败，`finish()` 会以非零状态退出。

阅读约定：每个函数的 docstring 都写了「参数（每个参数是什么、从哪来、取值）」「返回」「异常」。
"""
import hashlib
import json
from _stage import case, demo_settings, event, finish, report, write_json
from assistant.adapters import AdapterError
from assistant.orchestrator import Orchestrator
from assistant.recommendation import RecommendationAdapter


class TimeoutRemote:
    """故障注入桩：任何召回都抛 UPSTREAM_TIMEOUT。

    它是 RecommendationAdapter 的 remote 参数，用来模拟“模块一挂了”；
    正确行为是降级成离线热门召回，而不是把异常抛给调用方。
    """

    def recall(self, request):
        """被推荐适配器调用时，故意抛一个“上游超时”。

        参数：
            request (RecommendationRequest): 适配器传进来的推荐请求。
                这个假上游完全不看它——无论请求是什么，都装作上游超时。

        返回：
            不会返回。永远抛异常。

        异常：
            AdapterError(code='UPSTREAM_TIMEOUT')：错误码就是真实上游超时时会用的那个，
            所以推荐适配器会走和真实故障完全一样的降级分支。
        """
        raise AdapterError('UPSTREAM_TIMEOUT', '验收故障注入：上游超时，启用离线召回')


def digest(response):
    """把一次响应里「可复现的部分」算成 sha256 指纹，用来判断两次结果是否一致。

    参数：
        response (dict): `Orchestrator.handle()` 返回的完整响应。
            本函数只取其中两个字段：
            - `items`：商品列表（含分数、来源、证据）；
            - `context`：会话状态（约束、偏好、最近展示的商品）。
            不取 `trace` 和 `elapsed_ms`，因为那里有耗时，每次跑都不一样。

    返回：
        str: 64 位十六进制的 sha256 字符串。两次请求如果 items 和 context 完全相同，指纹就相同。

    说明：
        比较前用 `sort_keys=True` 排序，避免字典顺序不同导致误判。
    """
    canonical = {'items': response['items'], 'context': response['context']}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def run(settings):
    """跑阶段3 的 4 条验收用例，落盘证据，并返回用例列表。

    参数：
        settings (config.settings.Settings): 本次验收要用的配置。
            正常传 `_stage.demo_settings()`——它把商品目录固定成 18 件合成商品、
            清空上游地址，保证结果和本机环境无关。

    返回：
        list[dict]: 4 条用例，每条包含
            name（用例名）、input（输入事件）、expected_status（期望状态）、
            response（实际响应）、passed（是否通过）；
            「确定性复现」那条还额外带 second_response / sha256 / comparison。

    副作用：
        写文件 `outputs/stage03_demo.json`（内容 = 用例列表 + metrics + generated_at）。

    异常：
        SystemExit(1)：任何一条用例失败时由 `finish()` 抛出，保证「跑失败就是失败」，
        不会被 CI 或 README 里的命令误当成功。
    """
    # 正常链路用离线合成目录；降级链路额外注入一个必超时的假上游。
    app = Orchestrator(settings)
    degraded_app = Orchestrator(settings, RecommendationAdapter(settings, TimeoutRemote()))

    def normal_ok(response):
        """正常推荐的判定函数。

        参数：
            response (dict): 正常链路跑出来的响应（输入是「蓝色背包预算300以内」）。

        返回：
            bool: 同时满足以下全部才为 True：
            - 状态是 ok 且至少有一件商品；
            - 每件商品品类都是背包、颜色都是蓝色；
            - 每件商品都有价格且 ≤ 300；
            - 每件商品都有 sources（来源可追溯）；
            - 分数是降序的。
        """
        scores = [item['score'] for item in response['items']]
        return (response['status'] == 'ok' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all(item['price'] is not None and item['price'] <= 300 for item in response['items'])
                and all(item['sources'] for item in response['items'])
                and scores == sorted(scores, reverse=True))

    def empty_ok(response):
        """硬约束空集的判定函数。

        参数：
            response (dict): 输入是「耳机预算1以内」，预期全被硬约束拒掉。

        返回：
            bool: 同时满足才为 True：
            - items 为空列表；
            - 提示文案里含“调整”（告诉用户下一步该做什么）；
            - 会话约束原样保留 `{category: '耳机', budget_max: 1}`（证明没有偷偷放宽预算）。
        """
        return (response['items'] == [] and '调整' in response['message']
                and response['context']['constraints'] == {'category': '耳机', 'budget_max': 1})

    def degraded_ok(response):
        """上游超时降级的判定函数。

        参数：
            response (dict): 由注入了 `TimeoutRemote` 的编排器跑出来的响应。

        返回：
            bool: 同时满足才为 True：
            - 状态仍是 ok（降级不等于失败，用户仍能拿到结果）；
            - `degraded` 为 True，且第一条警告的错误码是 `UPSTREAM_TIMEOUT`；
            - 仍有商品，且品类/颜色仍符合用户条件（硬约束在降级后照样生效）；
            - 每件商品的来源里都有 `popularity`（证明走的是热门兜底）。
        """
        return (response['status'] == 'ok' and response['degraded']
                and response['warnings'][0]['code'] == 'UPSTREAM_TIMEOUT' and bool(response['items'])
                and all(item['category'] == '背包' and item['color'] == '蓝色' for item in response['items'])
                and all('popularity' in item['sources'] for item in response['items']))

    # 前三条：分别验证成功、边界（空集）、失败降级三类行为。
    cases = [case('正常推荐', app, event('orchestrator-normal', '蓝色背包预算300以内'), 'ok', normal_ok),
             case('硬约束空集', app, event('orchestrator-empty', '耳机预算1以内', session='empty'), 'empty', empty_ok),
             case('上游超时降级', degraded_app, event('orchestrator-timeout', '蓝色背包预算300以内'), 'ok', degraded_ok)]

    # 第四条：同一事件用两个全新 Orchestrator 各跑一次，比较 items+context 的 sha256。
    # 排序键是 (-score, item_id) 且没有随机数，所以两次结果必须逐字段一致。
    repro_event = event('repro', '推荐礼物预算300以内')
    first, second = Orchestrator(settings).handle(repro_event), Orchestrator(settings).handle(repro_event)
    first_digest, second_digest = digest(first), digest(second)
    same = (first_digest == second_digest and first['items'] == second['items']
            and first['context'] == second['context'])
    cases.append({'name': '确定性复现', 'input': repro_event, 'expected_status': first['status'],
                  'response': first, 'second_response': second, 'passed': bool(same),
                  'sha256': {'first': first_digest, 'second': second_digest, 'equal': first_digest == second_digest},
                  'comparison': {'same_items_and_scores': first['items'] == second['items'],
                                 'same_context': first['context'] == second['context']}})
    # 落盘 + 打印；finish 在有失败时抛 SystemExit(1)，保证“跑失败就是失败”。
    write_json(settings, 'stage03_demo.json', report(cases))
    finish(settings, cases, '阶段03')
    return cases


if __name__ == '__main__':
    run(demo_settings())
