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


def mediapipe_cache():
    """挑一个纯 ASCII 的可写目录给 MediaPipe 缓存用。
    例如项目路径含中文时 MediaPipe 会失败，因此要改用 ASCII 目录。

    为什么需要：worker 的 TEMP/TMP 被指到项目内的 runtime/module_two（Windows 上
    项目路径含中文），而 MediaPipe 加载 C++ 图资源时走窄字符接口，遇到中文路径会直接
    报错。所以这里显式给一个 ASCII 目录；写不进去就退回 C 盘根目录。
    """
    candidates = []
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    if local_app_data and str(local_app_data).isascii():
        candidates.append(Path(local_app_data) / 'mmai_mediapipe_cache')
    candidates.append(Path('C:/mmai_mediapipe_cache'))
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    return candidates[-1]


class ModuleTwoWorker:
    """按需拉起常驻识别进程，同一时刻只处理一个请求，避免管道错乱。
    例如首次识别才启动子进程，之后复用；并发请求排队等锁而不是交叉读写管道。
    """

    def __init__(self, settings):
        """保存配置并解析出模块二解释器路径，进程本身延迟到首次调用再启动。
        例如未配解释器路径时，默认取 .venv/Scripts/python.exe。

        参数：
            settings: 提供 module_two_python、module_two_root、超时与端口等配置。
        """
        self.settings = settings
        self.python = Path(settings.module_two_python) if settings.module_two_python else (
            settings.module_two_root / '.venv/Scripts/python.exe')
        self.process = self.responses = None
        self.lock = threading.RLock()
        self.closed = False
        atexit.register(self.close)

    def status(self):
        """报告模块二可用性：解释器、asr.py 与进程状态是否齐备。
        例如部署机没装模块二环境时返回 runtime_missing，前端据此提示改用文字输入。

        返回：
            dict: configured、running、state、vision_port 四个字段。
        """
        available = self.python.is_file() and (self.settings.module_two_root / 'src/asr.py').is_file()
        return {'configured': available, 'running': self.process is not None and self.process.poll() is None,
                'state': 'runtime_available' if available else 'runtime_missing',
                'vision_port': self.settings.module_two_vision_port}

    def _start(self):
        """拉起识别子进程，起线程把 stdout 的 JSON 行投递到队列。
        例如首次识别或进程意外退出后调用，端口、TEMP、缓存目录在这里一次性备好。
        """
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
                   IRA_WORKER_RUNTIME=str(runtime), TEMP=str(runtime), TMP=str(runtime),
                   # TEMP 指在含中文的项目路径下，MediaPipe 会拒绝其默认缓存目录，这里显式改到 ASCII 目录
                   MMAI_MEDIAPIPE_CACHE=str(mediapipe_cache()))
        self.process = subprocess.Popen(
            [str(self.python), '-B', '-u', str(ROOT / 'integrations/module_two_worker.py'),
             str(self.settings.module_two_root)],
            cwd=runtime, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.responses = queue.Queue(maxsize=8)
        responses, process = self.responses, self.process

        def read():
            """独占读取 stdout 入队；退出时投 None 让等待方失败。
            例如子进程崩溃后 call() 收到 None，直接按无效响应回收重来。
            """
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
        """发送请求并等待同 id 响应；异常统一抛 AdapterError。
        例如视觉识别超时会回收进程并提示可重试，而不是把半截结果当成成功。

        参数：
            operation (str): 操作名，如 voice、vision、gesture。
            data (dict): 传给模块二的数据。

        返回：
            dict: 模块二返回的 result 对象。
        """
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
        """优雅关停子进程：先发 shutdown，超时则按进程树强杀。
        例如 Windows 上用 taskkill /T /F 清理整棵进程树。
        """
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
        """标记关闭并回收进程，程序退出时由 atexit 自动调用一次。
        例如进程内重复调用也安全，第二次进来会因 process 已为 None 直接返回。
        """
        with self.lock:
            self.closed = True
            self._stop()
