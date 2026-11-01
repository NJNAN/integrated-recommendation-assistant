"""
课程基线 Web 服务（REST API + 页面）。启动：python -m baseline.web --port 5002

职责：
1. 加载基线引擎（engine.pkl）与精排模型（rankers.pkl），提供推荐、相似商品、
   健康检查、评估结果等 JSON 接口（详见 README_BASELINE.md「API」）；
2. 提供首页页面（templates/index.html）；
3. 扩展：/api/replenishment 补货提醒端点（第 5 路召回的在线展示，懒加载）。

设计约定：
- 统一 JSON 外层 {code, message, data}；错误分四级：参数错误 400、
  资源不存在 404、方法不允许 405、内部异常 500，全部走同一格式；
- 服务启动前校验 outputs/baseline/run_status.json 的 complete 标记，
  流水线未跑完时拒绝启动，避免把半成品模型当可用模型暴露；
- 引擎与精排模型只加载一次（Flask 应用配置项），每个请求复用；
  补货通道则按需懒加载（首次请求才从 history.csv 构建），失败降级 503。
"""
import argparse
import json
import pickle

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from threadpoolctl import threadpool_limits

from .config import MODELS, OUTPUTS
from .engine import BaselineEngine


def create_app(engine=None, rankers=None):
    """构建 Flask 应用；engine/rankers 可注入（测试用），缺省时从磁盘加载。

    参数
    ----
    engine : BaselineEngine | None
        注入的引擎。为 None 时检查 run_status 后从 models/baseline 加载。
    rankers : dict[str, model] | None
        注入的精排模型字典（'lr' 必需，可选 'lightgbm'）。为 None 时从
        rankers.pkl 加载。

    返回
    ----
    Flask 应用实例。测试代码可直接 create_app(engine, rankers) 注入
    内存中的小模型，避免依赖真实训练产物（见 baseline/tests）。
    """
    app = Flask(__name__)
    app.json.ensure_ascii = False  # 返回中文商品名时不转义成 \uXXXX
    if engine is None:
        # 状态机门禁：pipeline 没跑完（含中途失败）就不许启动 Web。
        status_path = OUTPUTS / 'run_status.json'
        if not status_path.exists() or not json.loads(status_path.read_text(encoding='utf-8')).get('complete'):
            raise RuntimeError('基线尚未构建完成，请先运行 python -m baseline.pipeline all')
        engine = BaselineEngine.load(MODELS)
        with (MODELS / 'rankers.pkl').open('rb') as f:
            rankers = pickle.load(f)
    app.config['ENGINE'] = engine
    app.config['RANKERS'] = rankers

    # 统一响应壳：HTTP 200 → code=0；否则 code=HTTP 状态码。
    def response(data, message='success', status=200):
        return jsonify({'code': 0 if status == 200 else status, 'message': message, 'data': data}), status

    # 解析 k/topn 参数：必须为 1~50 的整数，否则抛 ValueError → 400。
    def topn(default=10):
        value = request.args.get('k', request.args.get('topn', str(default)))
        try:
            k = int(value)
        except (TypeError, ValueError):
            raise ValueError('k/topn 必须是整数')
        if not 1 <= k <= 50:
            raise ValueError('k/topn 必须在 1～50 之间')
        return k

    @app.errorhandler(ValueError)
    def bad_request(error):
        """业务参数错误统一转 400（而不是 500）。"""
        return response(None, str(error), 400)

    @app.errorhandler(HTTPException)
    def http_error(error):
        """Werkzeug 内置错误（404/405 等）统一套上 JSON 外层。"""
        return response(None, error.name, error.code)

    @app.errorhandler(Exception)
    def internal_error(error):
        """兜底异常：详细堆栈只进服务端日志，不返回给客户端。"""
        app.logger.exception('baseline request failed')
        return response(None, '服务器内部错误', 500)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/health')
    @app.get('/api/health')
    def health():
        """健康检查：报告引擎商品数与可用精排模型列表。"""
        return response({'status': 'ok', 'items': len(engine.ids), 'models': list(rankers)})

    @app.get('/api/samples')
    def samples():
        """示例用户与商品：页面下拉框用（订单最多用户 + 最热门有向量商品）。"""
        users = sorted(engine.users, key=lambda u: (-engine.orders[u], u))[:2]
        items = [i for i in sorted(engine.ids, key=lambda i: (-engine.popularity[engine.pos[i]], i))
                 if engine.in_vocab[engine.pos[i]]][:2]
        return response({'users': users, 'items': items})

    @app.get('/similar')
    @app.get('/api/similar')
    def similar():
        """Item2Vec 相似商品查询；未知/不在词表商品返回 404。"""
        item = request.args.get('item_id', '').strip()
        if not item:
            raise ValueError('item_id 不能为空')
        k = topn(5)
        try:
            rows = engine.similar(item, k)
        except KeyError:
            return response(None, '商品不在 Item2Vec 词表中', 404)
        return response({'item_id': item, 'items': rows})

    @app.get('/api/recommendations')
    def recommend():
        """用户 Top-N 推荐。

        model 可选 popular/itemcf/已加载精排名（lr、lightgbm）：
        - 精排模型：engine.recommend 内部先四路召回融合出候选，再预测排序；
        - popular/itemcf：只展示该路的原始 Top-K（配额临时放大到 50，
          保证单路也能给出足够长的列表做演示），候选仍排除历史已购。
        冷启动用户（无历史）由引擎自然退化为热门候选，cold_start 字段标记。
        """
        uid = request.args.get('user_id', '').strip()
        model = request.args.get('model', 'lr')
        k = topn()
        if model not in {'popular', 'itemcf', *rankers}:
            raise ValueError('model 必须是 popular、itemcf 或已加载的精排模型名称')
        if model in rankers:
            rows = engine.recommend(uid, rankers[model], k)
        else:
            _, scores, routes = engine.candidates(uid, quotas={r: 50 for r in engine.cfg.quotas()})
            rows = [engine.enrich(i, scores[model][engine.pos[i]], [model]) for i in routes[model][:k]]
        return response({'user_id': uid, 'model': model, 'cold_start': uid not in engine.histories, 'items': rows})

    @app.get('/api/evaluations/latest')
    def evaluations():
        """返回最新离线评估摘要（evaluation.json），未评估过则 404。"""
        path = OUTPUTS / 'evaluation.json'
        if not path.exists():
            return response(None, '尚无评估结果', 404)
        return response(json.loads(path.read_text(encoding='utf-8')))

    # ---- 扩展：补货提醒（第5路召回的在线展示，懒加载，不影响既有端点）----
    _rep_route = {}

    def get_rep_route():
        """首次访问时基于 history.csv 构建补货通道；无数据时返回 None 降级。"""
        if 'route' not in _rep_route:
            try:
                import pandas as pd
                from .config import PROCESSED
                from .replenishment import ReplenishmentRoute
                hist = pd.read_csv(PROCESSED / 'history.csv',
                                   dtype={'user_id': str, 'item_id': str, 'invoice_no': str},
                                   parse_dates=['transaction_time'])
                _rep_route['route'] = ReplenishmentRoute(
                    hist, engine.pos, len(engine.ids), half_life_days=60.0, min_freq=2)
            except Exception:
                app.logger.exception('补货通道构建失败，降级为不可用')
                _rep_route['route'] = None
        return _rep_route['route']

    @app.get('/api/replenishment')
    def replenishment():
        """输入 user_id，返回该用户最可能需要补货的已购商品（RFM-lite 排序）。

        与 /api/recommendations 的区别：本端点【只返回已购商品】，面向批发补货场景；
        推荐端点排除已购、面向新品发现。两者构成课程报告中的"双轨"演示。
        """
        uid = request.args.get('user_id', '').strip()
        if not uid:
            raise ValueError('user_id 不能为空')
        k = topn()
        route = get_rep_route()
        if route is None:
            return response(None, '补货通道不可用：history.csv 缺失或构建失败', 503)
        if uid not in engine.histories:
            return response({'user_id': uid, 'cold_start': True, 'items': []},
                            '该用户无历史购买，无补货候选')
        idxs = route.recommend(uid, k)
        rows = [engine.enrich(engine.ids[i], float(route.score(uid)[i]), ['replenish'])
                for i in idxs]
        return response({'user_id': uid, 'cold_start': False, 'items': rows})

    return app


def main():
    """命令行启动：--port 指定端口（默认 5002），单线程限制保证复现。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5002)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        create_app().run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
