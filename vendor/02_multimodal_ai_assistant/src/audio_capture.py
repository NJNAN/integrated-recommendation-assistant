"""麦克风音频采集模块。

所属链路：语音输入链路（音频采集 → VAD 端点检测 → ASR 转写）。
职责：封装 PyAudio，将麦克风 PCM 流以"确定性生命周期"的方式暴露出来。
    - 输入：调用方传入采样率、chunk 大小、声道数、输入设备索引；
    - 输出：通过 :meth:`read_chunk` 反复产出固定字节长度的 PCM 字节串；
    - 调用方：``AudioStreamThread``、``scripts/01_capture_audio.py`` 等；
    - 依赖：可选懒加载 ``pyaudio``，缺失时给出明确的中文报错。

设计要点：
    1. 默认 16 kHz / 单声道 / 16-bit PCM，与下游 VAD（WebRTC VAD）和 SenseVoice 模型对齐；
    2. 通过 ``_lock`` 串行化 ``start`` / ``stop``，避免重复开关底层 PortAudio 流；
    3. ``stop`` 在 ``__exit__`` 兜底，确保 with 块退出时一定释放底层资源。
"""

from __future__ import annotations

import logging
import threading

LOGGER = logging.getLogger(__name__)


class AudioCapture:
    """16 kHz mono PCM 麦克风采集器，具备确定性生命周期。

    通过 ``with`` 语法或 ``start()`` / ``stop()`` 显式控制底层 PortAudio
    流的开关。重复调用 ``start()`` 是幂等的；``stop()`` 可在异常路径下
    安全调用。``backend`` 参数允许测试时注入假的 PyAudio 对象。
    """

    def __init__(
        self,
        rate: int = 16_000,
        chunk: int = 1_024,
        channels: int = 1,
        input_device_index: int | None = None,
        backend=None,
    ):
        """初始化采集器参数；并不立即打开设备。

        :param rate: 采样率（Hz），需与下游 VAD 支持的采样率集合匹配。
        :param chunk: 每次 ``read_chunk`` 返回的样本数，默认 1024。
        :param channels: 声道数，默认单声道（VAD / ASR 都按单声道处理）。
        :param input_device_index: 指定输入设备索引；``None`` 表示用系统默认麦克风。
        :param backend: 可选的 PyAudio 后端（测试桩）；``None`` 表示内部按需创建。
        """
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.input_device_index = input_device_index
        self.pa = backend
        self.stream = None
        # 标记后端是否由本对象创建，决定 stop() 时是否要 terminate 整个 PortAudio 实例。
        self._owns_backend = backend is None
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        """当前 PortAudio 输入流是否已启动且处于活动状态。"""
        return bool(self.stream and self.stream.is_active())

    def start(self) -> "AudioCapture":
        """打开麦克风流；幂等。

        若底层 PyAudio 不存在则抛出明确的 ``RuntimeError``；打开失败时
        自动回收本对象内部创建的后端实例，避免半成品状态泄漏。
        """
        with self._lock:
            if self.is_active:
                return self
            try:
                import pyaudio
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 PyAudio。请在项目虚拟环境运行 pip install PyAudio。"
                ) from exc
            if self.pa is None:
                self.pa = pyaudio.PyAudio()
            try:
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk,
                    input_device_index=self.input_device_index,
                )
            except Exception:
                # 打开失败时若后端由本对象创建，需主动 terminate 并清空引用，
                # 否则下次 start() 会拿到一个半死的 PyAudio 实例。
                if self._owns_backend and self.pa is not None:
                    self.pa.terminate()
                    self.pa = None
                raise
            LOGGER.info("麦克风已启动: %s Hz, %s 声道", self.rate, self.channels)
            return self

    def read_chunk(self) -> bytes:
        """读取一帧固定大小的 PCM 字节。

        流未启动时抛出 ``RuntimeError``；使用 ``exception_on_overflow=False``
        让 PortAudio 在环形缓冲区溢出时返回当前缓存的内容而不是抛异常，
        这样语音链路不会因为短暂的丢帧而中断。
        """
        if not self.is_active:
            raise RuntimeError("音频流尚未启动")
        return self.stream.read(self.chunk, exception_on_overflow=False)

    def stop(self) -> None:
        """关闭输入流并按需关闭底层 PyAudio 后端。

        使用 ``_lock`` 与 ``try/finally`` 保证即使 ``stop_stream`` 抛错
        也能继续关闭流对象与后端，避免资源泄漏。
        """
        with self._lock:
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                finally:
                    self.stream.close()
                    self.stream = None
            if self._owns_backend and self.pa is not None:
                self.pa.terminate()
                self.pa = None
            LOGGER.info("麦克风已停止")

    def __enter__(self) -> "AudioCapture":
        """进入 with 块时自动启动采集。"""
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出 with 块时自动停止采集（无论是否异常）。"""
        self.stop()