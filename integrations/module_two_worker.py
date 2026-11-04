"""薄入口：只在模块二自己的 Python 环境里跑的常驻进程，用 stdin/stdout 传 JSON。
例如收到 voice 请求才懒加载 ASR 模型，不预先占内存。
"""
import base64
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time

# Reserve stdout for IPC, including libraries that write directly to fd 1.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
runtime = Path(os.environ['IRA_WORKER_RUNTIME'])
asr = vision = detector = None
recognizers = {}
detector_key = None


def image_frame(encoded):
    """把 base64 图解码成 OpenCV 帧，先校验尺寸再占内存。
    例如超过 4096 边长或 400 万像素的图直接报错，不交给 OpenCV。
    """
    # Validate decompressed dimensions before OpenCV allocates a pixel array.
    from PIL import Image
    import cv2
    import numpy as np
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ('JPEG', 'PNG') or max(image.size) > 4096 or image.width * image.height > 4000000:
            raise ValueError('图片尺寸过大，请缩小到4096边长和400万像素以内')
        image.verify()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('无法解码图片')
    h, w = frame.shape[:2]
    if max(h, w) > 640:
        frame = cv2.resize(frame, (round(w * 640 / max(h, w)), round(h * 640 / max(h, w))))
    return frame


def infer(operation, data):
    """按操作类型分发到语音、视觉、手势模型，首次才加载。
    例如同一用户连续手势复用 detector，换用户就清空时序状态。
    """
    global asr, vision, detector, detector_key
    if operation == 'voice':
        if asr is None:
            from config.settings import MODEL_CONFIG
            from src.asr import ASREngine
            model = Path(MODEL_CONFIG['asr'])
            if not (model / 'model.pt').is_file():
                raise RuntimeError('本地 SenseVoice 模型缺失；不自动联网下载')
            # CPU keeps ASR resident without competing with the vision model's VRAM.
            asr = ASREngine(str(model), device='cpu')
        transcript = asr.transcribe(base64.b64decode(data['audio'], validate=True))
        if not transcript.strip():
            raise ValueError('未识别到有效语音，请重录或填写转写文本')
        return {'transcript': transcript, 'model': 'SenseVoiceSmall', 'device': asr.device}
    frame = image_frame(data['image'])
    if operation == 'vision':
        if vision is None:
            from config.settings import LLAMA_CONFIG
            from copy import deepcopy
            config = deepcopy(LLAMA_CONFIG)
            config['log_dir'] = runtime
            config['request_timeout'] = 150
            config['start_timeout'] = 120
            config['vision']['disable_thinking'] = True
            # Refuse occupied ports before module two's legacy reclaim logic can run.
            port = config['vision']['port']
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError as exc:
                    raise RuntimeError(f'视觉端口 {port} 已占用，请设置 IRA_MODULE_TWO_VISION_PORT') from exc
            from src.llama_backend import create_vision_client
            from src.vl_model import VLModel
            vision = VLModel(config['vision']['alias'], client=create_vision_client(config))
        # Recheck after idle shutdown, so an occupied port is never reclaimed from another application.
        if not vision.client.manager.any_running():
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', vision.client.manager.describe()['vision']['port']))
                except OSError as exc:
                    raise RuntimeError('视觉端口已占用，请调整 IRA_MODULE_TWO_VISION_PORT') from exc
        prompt = ('分别识别图中唯一主要商品的品类和主体颜色，两个都要识别。只输出JSON，不输出思考、解释或Markdown。'
                  '格式例子：{"labels":[{"label":"杯子","confidence":0.9},{"label":"红色","confidence":0.9}]}。'
                  '品类只允许：背包、杯子、耳机、台灯、键盘、雨伞；颜色只允许：蓝色、红色、绿色、黑色、白色。'
                  '最多一个品类、一个颜色。无法确认或没有相关商品时输出{"labels":[]}。不要猜测价格、品牌或用途。')
        output = vision.analyze_frames([frame], prompt, max_new_tokens=160)
        # Import the integration parser without importing project three's config package.
        spec = importlib.util.spec_from_file_location('ira_media', Path(__file__).parents[1] / 'assistant/media.py')
        # Parser dependencies live in project three; append so module two's config stays first.
        integration_root = str(Path(__file__).parents[1])
        if integration_root not in sys.path:
            sys.path.append(integration_root)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {'labels': module.parse_visual_labels(output), 'model': vision.model_name}
    if operation == 'gesture':
        from src.gesture import GestureRecognizer
        from src.human_detector import HumanDetector
        key = tuple(data.get(k, '') for k in ('user_id', 'session_id', 'stream_id', 'item_id'))
        if detector_key != key:
            if detector is not None:
                detector.close()
            detector = HumanDetector(model_complexity=1)
            detector_key = key
            # Reset temporal state when switching users, stream, or feedback target.
            recognizers.clear()
        recognizer = recognizers.setdefault(key, GestureRecognizer())
        hands = detector.process_frame(frame)['hands']
        landmarks = getattr(hands, 'multi_hand_landmarks', None)
        # Empty hand input releases a held gesture, allowing the next confirmation.
        gesture = recognizer.recognize(landmarks[0] if landmarks else None, aspect_ratio=frame.shape[1] / frame.shape[0])
        return {'gesture': gesture, 'hand_detected': bool(landmarks), 'model': 'MediaPipe + GestureRecognizer'}
    raise ValueError('不支持的操作')


try:
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get('operation') == 'shutdown':
                break
            start = time.monotonic()
            result = infer(request['operation'], request['data'])
            result['elapsed_ms'] = round((time.monotonic() - start) * 1000)
            response = {'id': request['id'], 'result': result}
        except Exception as exc:
            response = {'id': request.get('id'), 'error': {'code': getattr(exc, 'code', 'INVALID_MEDIA' if isinstance(exc, ValueError) else 'UPSTREAM_UNAVAILABLE'),
                                                        'message': str(exc)[:500]}}
        protocol.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n')
        protocol.flush()
finally:
    if detector is not None:
        detector.close()
    if 'src.llama_backend' in sys.modules:
        sys.modules['src.llama_backend'].shutdown()
