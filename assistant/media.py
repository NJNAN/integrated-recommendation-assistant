"""媒体边界：限制上传 -> 调模块二识别 -> 组成标准事件 -> 交给编排器。"""
import base64
import binascii
from datetime import datetime, timezone
import io
import json
import re
import wave
from assistant.contracts import ContractError, InteractionEvent, nonempty, number

VISUAL_LABELS = {'背包', '杯子', '耳机', '台灯', '键盘', '雨伞', '蓝色', '红色', '绿色', '黑色', '白色'}
# 模型可能在 JSON 外面包一层 <标签>…</标签>（例如思考过程），解析前先去掉。
_WRAPPER = re.compile(r'<[^<>]{1,20}>.*?</[^<>]{1,20}>', re.S)
_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$')


def decode_media(encoded, limit):
    """Base64 解码并校验大小；任何不合法输入都以 INVALID_MEDIA 明确失败。"""
    if not isinstance(encoded, str) or len(encoded) > limit * 4 // 3 + 4:
        raise ContractError('媒体数据缺失或超过大小限制', 'INVALID_MEDIA')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ContractError('媒体应为合法 Base64 字符串', 'INVALID_MEDIA') from exc
    if not raw or len(raw) > limit:
        raise ContractError('媒体不能为空或超过大小限制', 'INVALID_MEDIA')
    return raw


def audio_pcm(encoded, fmt):
    """返回 16kHz/16bit/单声道 PCM；WAV 头必须匹配，长度限制 0.1~20 秒。"""
    raw = decode_media(encoded, 700000)
    if fmt == 'wav':
        try:
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                header = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                if header != (1, 2, 16000, 'NONE'):
                    raise ValueError
                raw = wav.readframes(wav.getnframes())
        except (ValueError, wave.Error, EOFError) as exc:
            raise ContractError('WAV 应为16kHz、16位、单声道 PCM，最长20秒', 'INVALID_MEDIA') from exc
    elif fmt != 'pcm16':
        raise ContractError('音频仅支持 WAV 或16kHz单声道 pcm16', 'INVALID_MEDIA')
    if not 3200 <= len(raw) <= 640000 or len(raw) % 2:
        raise ContractError('录音应为0.1到20秒的16位 PCM', 'INVALID_MEDIA')
    return raw


def parse_visual_labels(text):
    """只接受 {"labels":[{label,confidence}]}；去掉模型包裹与代码围栏，拒绝自由文本。"""
    if not isinstance(text, str):
        raise ContractError('视觉模型没有返回结构化标签', 'INVALID_PERCEPTION')
    cleaned = _FENCE.sub('', _WRAPPER.sub('', text).strip()).strip()
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise ContractError('视觉输出不是合法 JSON，请改用文字或手动标签', 'INVALID_PERCEPTION') from exc
    labels = data.get('labels') if isinstance(data, dict) else None
    if not isinstance(labels, list) or len(labels) > 2:
        raise ContractError('视觉输出应包含最多两个标签', 'INVALID_PERCEPTION')
    result, kinds = [], set()
    for tag in labels:
        if not isinstance(tag, dict) or tag.get('label') not in VISUAL_LABELS:
            raise ContractError('视觉标签不在支持的品类或颜色内', 'INVALID_PERCEPTION')
        kind = 'color' if tag['label'].endswith('色') else 'category'
        if kind in kinds:
            raise ContractError('视觉输出包含相互冲突的标签', 'INVALID_PERCEPTION')
        kinds.add(kind)
        confidence = number(tag.get('confidence'), 'confidence', 0, 1)
        result.append({'label': tag['label'], 'confidence': confidence})
    return result


class MediaService:
    """媒体入口：解码校验 -> worker 识别 -> 统一事件 -> 编排，并在响应里附带 perception。"""

    def __init__(self, orchestrator, worker):
        self.app, self.worker = orchestrator, worker

    def handle(self, source, data):
        if not isinstance(data, dict):
            raise ContractError('媒体请求必须是 JSON 对象')
        ids = {key: nonempty(data.get(key), key) for key in ('trace_id', 'user_id', 'session_id')}
        if set(data) - (set(ids) | {'audio', 'format', 'image', 'text', 'item_id', 'stream_id'}):
            raise ContractError('媒体请求包含未知字段')
        payload = {'text': nonempty(data['text'], 'text')} if data.get('text') else {}
        if source == 'voice':
            pcm = audio_pcm(data.get('audio'), data.get('format'))
            perceived = self.worker.call('voice', {**ids, 'audio': base64.b64encode(pcm).decode()})
            payload['transcript'] = nonempty(perceived.get('transcript'), 'transcript')
        elif source in ('vision', 'gesture'):
            image = decode_media(data.get('image'), 2 * 1024 * 1024)
            if not (image.startswith(b'\xff\xd8\xff') or image.startswith(b'\x89PNG\r\n\x1a\n')):
                raise ContractError('图片仅支持 JPEG 或 PNG', 'INVALID_MEDIA')
            extra = {key: nonempty(data[key], key) for key in ('stream_id', 'item_id') if data.get(key)}
            perceived = self.worker.call(source, {**ids, **extra, 'image': data['image']})
            if source == 'vision':
                raw_labels = json.dumps({'labels': perceived.get('labels')}, allow_nan=False)
                payload['labels'] = parse_visual_labels(raw_labels)
            elif perceived.get('gesture') not in ('thumbs_up', 'thumbs_down', 'open_palm', 'victory'):
                return {'status': 'pending', **ids, 'perception': perceived, 'message': '等待稳定的支持手势'}
            else:
                payload['gesture'] = perceived['gesture']
                if data.get('item_id'):
                    payload['item_id'] = data['item_id']
        else:
            raise ContractError('不支持的媒体类型')
        event = InteractionEvent.from_dict({
            'trace_id': ids['trace_id'], 'user_id': ids['user_id'], 'session_id': ids['session_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(), 'source': source, 'payload': payload})
        result = self.app.handle(event.to_dict())
        result['perception'] = {**perceived, 'provider': 'module_two', 'trace_id': ids['trace_id']}
        if source == 'vision':
            result['perception']['confidence_kind'] = 'model_self_reported'
        return result
