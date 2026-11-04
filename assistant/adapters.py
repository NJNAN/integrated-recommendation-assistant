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
    """承载外部服务失败的异常，错误码告诉调用方该重试还是降级。

    例如 UPSTREAM_TIMEOUT 可重试，UPSTREAM_USER_MISMATCH 只能提示换账号。

    参数：
        code (str): 机器可读错误码。
        message (str): 给人看的失败原因。
    """

    def __init__(self, code, message):
        """保存错误码和原因，交给上层决定重试、降级还是提示用户。

        例如 code='UPSTREAM_TIMEOUT' 时编排器会走一次降级。

        参数：
            code (str): 机器可读错误码。
            message (str): 给人看的失败原因。
        """
        super().__init__(message)
        self.code = code


class MultimodalAdapter:
    """把已转写的语音、已识别的视觉标签和已确认的手势包装成统一事件。

    例如视觉链路只传 labels，这里补齐追踪字段再交给契约校验。
    """

    def event(self, source, payload, user_id, session_id, trace_id=None):
        """包装成统一事件：补齐追踪字段和时间戳后立刻过一遍契约校验。

        例如语音链路只给 transcript，这里自动生成 trace_id 和 UTC 时间。

        参数：
            source (str): 输入来源，text/voice/vision/gesture 之一。
            payload (dict): 交互内容，字段受契约白名单限制。
            user_id (str): 用户标识。
            session_id (str): 会话标识。
            trace_id (str | None): 已有追踪 ID，缺省时自动生成。

        返回：
            dict: 可直接 JSON 序列化的事件字典。

        异常：
            ContractError: 来源或内容不合法时抛出。
        """
        return InteractionEvent.from_dict({
            'trace_id': trace_id or str(uuid.uuid4()), 'user_id': user_id, 'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'source': source, 'payload': payload,
        }).to_dict()


class HTTPRecommender:
    """调用模块一冻结 Web API 召回候选，每次请求用独立 CookieJar 防串号。

    例如同时演示两个账号时，各自的登录态不会互相覆盖。
    """

    def __init__(self, settings):
        """记住配置，召回时从这里取地址、账号和超时。

        例如离线模式下 recommender_url 为空，编排器不会调用召回。

        参数：
            settings (Settings): 全局配置。
        """
        self.settings = settings

    def recall(self, request):
        """登录模块一并分页拉候选，返回原始商品行；失败一律转成 AdapterError。

        例如 user_id 与登录账号对不上时抛 UPSTREAM_USER_MISMATCH，让上层提示换用户。

        参数：
            request (RecommendationRequest): 已通过契约校验的推荐请求。

        返回：
            list[dict]: 上游返回的商品行。

        异常：
            AdapterError: 登录失败、超时、响应非法或候选超上限时抛出。
        """
        opener = build_opener(HTTPCookieProcessor(CookieJar()))

        def call(path, body=None):
            """发一次请求并解包 {code:0, data:...}，各类失败都映射成带码异常。

            例如 HTTP 500 变 UPSTREAM_UNAVAILABLE，超时变 UPSTREAM_TIMEOUT。

            参数：
                path (str): 相对接口路径，如 /api/auth/login。
                body (dict | None): 请求体，GET 时传 None。

            返回：
                object: 上游 data 字段的内容。

            异常：
                AdapterError: 网络、状态码或 JSON 解包失败时抛出。
            """
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
