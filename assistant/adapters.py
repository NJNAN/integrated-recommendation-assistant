"""外部适配边界：多模态事件构造与模块一 HTTP 召回；失败必须带错误码。"""
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener
import json
import socket
import uuid
from assistant.contracts import InteractionEvent


class AdapterError(RuntimeError):
    """带错误码的外部服务异常：调用方据此决定降级策略。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class MultimodalAdapter:
    """把已转写的语音、已识别的视觉标签和已确认的手势包装成统一事件。"""

    def event(self, source, payload, user_id, session_id, trace_id=None):
        return InteractionEvent.from_dict({
            'trace_id': trace_id or str(uuid.uuid4()), 'user_id': user_id, 'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'source': source, 'payload': payload,
        }).to_dict()


class HTTPRecommender:
    """模块一冻结 Web API：每次召回独立 CookieJar，避免用户身份串线。"""

    def __init__(self, settings):
        self.settings = settings

    def recall(self, request):
        opener = build_opener(HTTPCookieProcessor(CookieJar()))

        def call(path, body=None):
            """发一次请求并解包 {code:0,data:...}，把各种失败映射成带码异常。"""
            raw = None if body is None else json.dumps(body).encode('utf-8')
            req = Request(self.settings.recommender_url + path, data=raw,
                          headers={'Content-Type': 'application/json', 'X-Trace-ID': request.trace_id})
            try:
                with opener.open(req, timeout=self.settings.upstream_timeout_seconds) as response:
                    encoded = response.read(self.settings.max_body_bytes + 1)
            except (TimeoutError, socket.timeout) as exc:
                raise AdapterError('UPSTREAM_TIMEOUT', '推荐接口超时') from exc
            except HTTPError as exc:
                raise AdapterError('UPSTREAM_UNAVAILABLE', f'推荐接口 HTTP {exc.code}') from exc
            except URLError as exc:
                timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
                raise AdapterError('UPSTREAM_TIMEOUT' if timeout else 'UPSTREAM_UNAVAILABLE',
                                   '推荐接口不可用') from exc
            if len(encoded) > self.settings.max_body_bytes:
                raise AdapterError('INVALID_UPSTREAM', '上游响应超过大小限制')
            try:
                result = json.loads(encoded)
            except ValueError as exc:
                raise AdapterError('INVALID_UPSTREAM', '推荐接口响应不是 JSON') from exc
            if not isinstance(result, dict) or result.get('code') != 0:
                raise AdapterError('INVALID_UPSTREAM', '上游返回错误响应')
            return result['data']

        username, password = self.settings.recommender_username, self.settings.recommender_password
        if self.settings.recommender_accounts:
            account = next((a for a in self.settings.recommender_accounts if a[1] == request.user_id), None)
            if account is None:
                raise AdapterError('UPSTREAM_USER_MISMATCH', '请使用已接入的模块一演示用户')
            username, password = account[0], account[3]
        identity = call('/api/auth/login', {'username': username, 'password': password})
        if not isinstance(identity, dict) or str(identity.get('user_id')) != request.user_id:
            raise AdapterError('UPSTREAM_USER_MISMATCH', '上游登录用户与请求 user_id 不一致')

        rows, cursor = [], 0
        while len(rows) < self.settings.recommender_max_candidates:
            query = urlencode({'limit': self.settings.max_top_k, 'cursor': cursor})
            data = call('/api/recommendations?' + query)
            if not isinstance(data, dict) or not isinstance(data.get('items'), list):
                raise AdapterError('INVALID_UPSTREAM', '推荐接口缺少 items 数组')
            rows.extend(data['items'])
            if not data.get('has_more'):
                return rows
            next_cursor = data.get('next_cursor')
            if type(next_cursor) is not int or next_cursor <= cursor:
                raise AdapterError('INVALID_UPSTREAM', '推荐分页游标没有前进')
            cursor = next_cursor
        raise AdapterError('INVALID_UPSTREAM', '推荐候选超过配置上限，请调整上限')
