"""模块二子进程适配：JSON-lines IPC，模型依赖留在模块二自己的虚拟环境里。"""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from config.settings import ROOT
from assistant.adapters import AdapterError


class ModuleTwoWorker:
    """按需拉起常驻识别进程；同一时刻只允许一个请求，保证 stdin/stdout 不串行错乱。"""

    def __init__(self, settings):
        self.settings = settings
        self.python = Path(settings.module_two_python) if settings.module_two_python else (
            settings.module_two_root / '.venv/Scripts/python.exe')
        self.process = self.responses = None
        self.lock = threading.RLock()
        self.closed = False
        atexit.register(self.close)

    def status(self):
        available = self.python.is_file() and (self.settings.module_two_root / 'src/asr.py').is_file()
        return {'configured': available, 'running': self.process is not None and self.process.poll() is None,
                'state': 'runtime_available' if available else 'runtime_missing',
                'vision_port': self.settings.module_two_vision_port}

    def _start(self):
        if self.closed:
            raise AdapterError('UPSTREAM_UNAVAILABLE', '多模态进程已关闭')
        if not self.status()['configured']:
            raise AdapterError('UPSTREAM_UNAVAILABLE',
                               '未找到模块二环境，请配置 IRA_MODULE_TWO_ROOT / IRA_MODULE_TWO_PYTHON')
        runtime = ROOT / 'runtime/module_two'
        runtime.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
                   MMAI_TEXT_PORT=str(self.settings.module_two_text_port),
                   MMAI_VL_PORT=str(self.settings.module_two_vision_port),
                   IRA_WORKER_RUNTIME=str(runtime), TEMP=str(runtime), TMP=str(runtime))
        self.process = subprocess.Popen(
            [str(self.python), '-B', '-u', str(ROOT / 'integrations/module_two_worker.py'),
             str(self.settings.module_two_root)],
            cwd=runtime, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.responses = queue.Queue(maxsize=8)
        responses, process = self.responses, self.process

        def read():
            """独占读取 stdout 投递到队列；进程退出时投递 None，让等待方立刻失败。"""
            try:
                for line in process.stdout:
                    try:
                        responses.put_nowait(json.loads(line))
                    except ValueError:
                        continue
            finally:
                try:
                    responses.put_nowait(None)
                except queue.Full:
                    pass

        threading.Thread(target=read, daemon=True).start()

    def call(self, operation, data):
        """发一条请求并等待同 id 的响应；超时、进程退出、上游报错都抛 AdapterError。"""
        deadline = time.monotonic() + self.settings.module_two_timeout
        if not self.lock.acquire(timeout=self.settings.module_two_timeout):
            raise AdapterError('UPSTREAM_BUSY', '多模态模型忙，请稍后重试')
        try:
            if self.process is None or self.process.poll() is not None:
                self._stop()
                self._start()
            request_id = str(uuid.uuid4())
            try:
                self.process.stdin.write(json.dumps({'id': request_id, 'operation': operation, 'data': data},
                                                    ensure_ascii=False, allow_nan=False) + '\n')
                self.process.stdin.flush()
                response = self.responses.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty as exc:
                self._stop()
                raise AdapterError('UPSTREAM_TIMEOUT', '多模态识别超时，进程已回收；可重试或使用手动输入') from exc
            except (BrokenPipeError, OSError) as exc:
                self._stop()
                raise AdapterError('UPSTREAM_UNAVAILABLE', '多模态进程连接失败') from exc
            if not isinstance(response, dict) or response.get('id') != request_id:
                self._stop()
                raise AdapterError('INVALID_UPSTREAM', '多模态进程退出或返回无效响应')
            if 'error' in response:
                error = response['error']
                raise AdapterError(error.get('code', 'UPSTREAM_UNAVAILABLE'), error.get('message', '识别失败'))
            if not isinstance(response.get('result'), dict):
                raise AdapterError('INVALID_UPSTREAM', '多模态结果不是对象')
            return response['result']
        finally:
            self.lock.release()

    def _stop(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write('{"operation":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                # 只结束本实例创建的进程树，避免视觉子进程继续占用端口。
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                else:
                    process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()

    def close(self):
        with self.lock:
            self.closed = True
            self._stop()
