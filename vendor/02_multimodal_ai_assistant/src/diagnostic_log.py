"""把业务 INFO 事件投递给独立系统信息框，原有文件日志继续保留。"""
import logging
import time


class DiagnosticLogHandler(logging.Handler):
    CHANNELS = {'src.rag': '知识库', 'src.hotwords': '热词', 'src.asr': '语音识别',
                'src.assistant_service': '会话/协同', 'src.audio_thread': '语音输入',
                'src.tts_player': '朗读'}

    def __init__(self, callback):
        super().__init__(logging.INFO)
        self.callback = callback
        self._levels = {}
        for name in self.CHANNELS:
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(self)

    def emit(self, record):
        try:
            message = record.getMessage()
            category = '热词' if '热词' in message else self.CHANNELS.get(record.name, '系统')
            self.callback({'time': time.strftime('%H:%M:%S', time.localtime(record.created)),
                           'category': category, 'message': message[:1600], 'level': record.levelname})
        except Exception:
            self.handleError(record)

    def close(self):
        for name, level in self._levels.items():
            logger = logging.getLogger(name)
            logger.removeHandler(self)
            logger.setLevel(level)
        self._levels.clear()
        super().close()
