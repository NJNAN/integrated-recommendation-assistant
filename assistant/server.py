"""本地 HTTP 边界：只监听 127.0.0.1，静态资源走白名单，响应带安全头且不记录请求体。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import uuid
from urllib.parse import urlsplit

from assistant.adapters import AdapterError
from assistant.contracts import ContractError
from assistant.media import MediaService
from assistant.module_two import ModuleTwoWorker
from assistant.orchestrator import Orchestrator

# 页面只用到这三个静态文件；media.js 已合并进 app.js，不再开放。
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8')}
MEDIA = {'/api/media/voice': 'voice', '/api/media/vision': 'vision', '/api/media/gesture': 'gesture'}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; object-src 'none'; frame-ancestors 'none'")


def _reject_constant(value):
    """拒绝 NaN/Infinity，避免非有限数字进入业务层。"""
    raise ValueError('JSON 不允许非有限数字')


def create_server(settings, orchestrator=None, media_worker=None):
    """组装线程化服务；orchestrator/worker 可注入，便于测试或替换上游。"""
    app = orchestrator or Orchestrator(settings)
    worker = media_worker or ModuleTwoWorker(settings)
    media = MediaService(app, worker)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def server_close(self):
            closer = getattr(worker, 'close', None)
            if closer:
                closer()
            super().server_close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 用户输入与请求体不进入访问日志

        def send_body(self, status, body, content_type='application/json; charset=utf-8'):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', CSP)
            self.end_headers()
            self.wfile.write(body)

        def fail(self, status, code, message, trace_id=''):
            """边界错误也带 trace_id，方便前端日志与后端响应对齐。"""
            self.send_body(status, {'trace_id': trace_id or str(uuid.uuid4()), 'status': 'error',
                                    'error': {'code': code, 'message': message}, 'message': message,
                                    'items': [], 'degraded': False, 'warnings': []})

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/health':
                catalog = app.recommender.catalog
                return self.send_body(200, {
                    'status': 'ok',
                    'mode': 'hybrid_configured' if settings.recommender_url else 'offline_demo',
                    'dataset': catalog.dataset, 'catalog_size': len(catalog.items),
                    'module_two': worker.status(), 'currency': catalog.currency,
                    'price_notice': catalog.price_notice,
                    'users': [{'user_id': row[1], 'name': row[2]} for row in settings.recommender_accounts],
                })
            if path in STATIC:
                name, content_type = STATIC[path]
                try:
                    return self.send_body(200, (settings.web_dir / name).read_bytes(), content_type)
                except OSError:
                    return self.fail(404, 'NOT_FOUND', '页面资源不存在')
            self.fail(404, 'NOT_FOUND', '接口或资源不存在')

        def do_POST(self):
            self.connection.settimeout(10)
            path = urlsplit(self.path).path
            if path != '/api/events' and path not in MEDIA:
                return self.fail(404, 'NOT_FOUND', '接口不存在')
            origin, host = self.headers.get('Origin'), self.headers.get('Host', '')
            if origin and origin not in ('http://' + host, 'https://' + host):
                return self.fail(403, 'INVALID_ORIGIN', '仅允许当前站点提交请求')
            if self.headers.get_content_type() != 'application/json':
                return self.fail(415, 'INVALID_CONTENT_TYPE', '请求应使用 application/json')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0:
                    return self.fail(400, 'INVALID_JSON', '请求体不能为空')
                limit = settings.max_media_body_bytes if path in MEDIA else settings.max_body_bytes
                if length > limit:
                    self.close_connection = True
                    return self.fail(413, 'BODY_TOO_LARGE', '请求体超过限制')
                data = json.loads(self.rfile.read(length), parse_constant=_reject_constant)
            except (ValueError, UnicodeDecodeError, TimeoutError):
                return self.fail(400, 'INVALID_JSON', '无法解析 JSON 请求')
            trace_id = data.get('trace_id') if isinstance(data, dict) and isinstance(
                data.get('trace_id'), str) else ''
            try:
                result = media.handle(MEDIA[path], data) if path in MEDIA else app.handle(data)
            except ContractError as exc:
                return self.fail(400, exc.code, str(exc), trace_id)
            except AdapterError as exc:
                if exc.code == 'UPSTREAM_TIMEOUT':
                    status = 504
                elif exc.code in ('INVALID_MEDIA', 'INVALID_PERCEPTION'):
                    status = 400
                else:
                    status = 503
                return self.fail(status, exc.code, str(exc), trace_id)
            except Exception:
                return self.fail(500, 'INTERNAL_ERROR', '系统处理失败，请重试', trace_id)
            if result.get('status') == 'error':
                # 业务错误照旧 400；只有内部异常才映射到 500。
                status = 500 if (result.get('error') or {}).get('code') == 'INTERNAL_ERROR' else 400
                return self.send_body(status, result)
            self.send_body(200, result)

    return Server((settings.host, settings.port), Handler)
