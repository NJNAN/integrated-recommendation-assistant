"""反馈语义与 HTTP 边界：覆盖不累加、非法拒绝、会话隔离和服务安全行为。"""
import json
import socket
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from assistant.adapters import AdapterError
from assistant.contracts import ContractError
from assistant.feedback import apply_feedback
from assistant.orchestrator import Orchestrator
from assistant.server import create_server
from assistant.session import SessionContext
from config.settings import Settings, offline_demo_settings

# 1x1 PNG：足够通过格式校验，但注入的 worker 不可用，用于验证 503 映射。
PNG = ('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA'
       'hKmMIQAAAABJRU5ErkJggg==')
EVENT = {'trace_id': 'http-1', 'user_id': 'user-1', 'session_id': 'session-1',
         'timestamp': '2026-10-08T10:00:00+08:00', 'source': 'text', 'payload': {'text': '推荐礼物'}}


class _NoWorker:
    """模拟未配置的模块二：识别请求明确返回 503，而不是空结果。"""

    def status(self):
        return {'configured': False, 'running': False, 'state': 'runtime_missing', 'vision_port': 0}

    def call(self, operation, data):
        raise AdapterError('UPSTREAM_UNAVAILABLE', '多模态服务未配置')

    def close(self):
        pass


def _session(last_ids=('demo-001', 'demo-002')):
    return SessionContext(user_id='u-1', session_id='s-1', last_item_ids=list(last_ids))


class FeedbackTests(unittest.TestCase):
    def test_repeat_feedback_overwrites_without_accumulating(self):
        session = _session()
        self.assertEqual(apply_feedback(session, 'demo-001', 1), 1)
        apply_feedback(session, 'demo-001', 1)
        self.assertEqual(session.preferences, {'demo-001': 1})
        apply_feedback(session, 'demo-001', -1)
        self.assertEqual(session.preferences, {'demo-001': -1})

    def test_invalid_sign_and_unseen_item_rejected(self):
        session = _session()
        for sign in (0, 2, -2, None, '1'):
            with self.subTest(sign=sign), self.assertRaises(ContractError) as ctx:
                apply_feedback(session, 'demo-001', sign)
            self.assertEqual(ctx.exception.code, 'INVALID_FEEDBACK')
        with self.assertRaises(ContractError) as ctx:
            apply_feedback(session, 'demo-999', 1)
        self.assertEqual(ctx.exception.code, 'INVALID_FEEDBACK')
        self.assertEqual(session.preferences, {})

    def test_sessions_are_isolated(self):
        first, second = _session(), _session()
        apply_feedback(first, 'demo-001', 1)
        self.assertEqual(first.preferences, {'demo-001': 1})
        self.assertEqual(second.preferences, {})


class HttpBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        active = offline_demo_settings(Settings(port=0))
        cls.server = create_server(active, orchestrator=Orchestrator(active), media_worker=_NoWorker())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port
        cls.base = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def call(self, path, raw=None, headers=None):
        request = Request(self.base + path, data=raw, headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self):
        status, body = self.call('/api/health')
        self.assertEqual(status, 200)
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['mode'], 'offline_demo')
        self.assertGreater(body['catalog_size'], 0)
        self.assertFalse(body['module_two']['configured'])

    def test_event(self):
        status, body = self.call('/api/events', json.dumps(EVENT).encode(),
                                 {'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        self.assertEqual(body['trace_id'], 'http-1')
        self.assertEqual(body['status'], 'ok')

    def test_unknown_path_404(self):
        self.assertEqual(self.call('/api/nope')[0], 404)

    def test_wrong_content_type_415(self):
        status, body = self.call('/api/events', b'{}', {'Content-Type': 'text/plain'})
        self.assertEqual(status, 415)
        self.assertEqual(body['error']['code'], 'INVALID_CONTENT_TYPE')

    def test_invalid_json_400(self):
        status, body = self.call('/api/events', b'{', {'Content-Type': 'application/json'})
        self.assertEqual(status, 400)
        self.assertEqual(body['error']['code'], 'INVALID_JSON')

    def test_oversized_body_413(self):
        with socket.create_connection(('127.0.0.1', self.port), timeout=5) as sock:
            head = (f'POST /api/events HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n'
                    'Content-Type: application/json\r\nContent-Length: 70000\r\n\r\n')
            sock.sendall(head.encode())
            status_line = sock.recv(256).split(b'\r\n', 1)[0]
        self.assertIn(b' 413 ', status_line)

    def test_media_without_worker_503(self):
        body = {'trace_id': 'm-1', 'user_id': 'u-1', 'session_id': 's-1', 'image': PNG}
        status, payload = self.call('/api/media/gesture', json.dumps(body).encode(),
                                    {'Content-Type': 'application/json'})
        self.assertEqual(status, 503)
        self.assertEqual(payload['error']['code'], 'UPSTREAM_UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
