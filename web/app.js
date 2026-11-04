'use strict';
// 拾物前端：文字 / 语音 / 视觉 / 手势四种输入 + 推荐卡片、反馈、比较、撤销与证据导出。
// 旧 app.js 与 media.js 合并为一个文件，只调用 /api/health、/api/events、/api/media/*。
const $ = (id) => document.getElementById(id);
const state = {mode: 'text', busy: false, recording: false, session: crypto.randomUUID(), user: 'web-demo',
  response: null, selected: new Set()};
const SLOT = {category: '品类', color: '颜色', budget_min: '最低预算', budget_max: '最高预算',
  recipient: '送给', scene: '场景', currency: '币种'};
const COLOR_CLASS = {'蓝色': 'blue', '红色': 'red', '绿色': 'green', '黑色': 'black', '白色': 'white'};
const ICON = {'背包': '🎒', '杯子': '🥤', '耳机': '🎧', '台灯': '💡', '键盘': '⌨️', '雨伞': '☂️'};
const HINT = {text: '支持多轮修改与撤销', voice: '录音识别最长20秒',
  vision: '识别品类与颜色，低置信标签会被忽略', gesture: '点赞 / 不喜欢需要先选择反馈商品'};
let recording = null, camera = null, cameraTimer = null, cameraStreamId = null; let recordingStarting = false, cameraStarting = false;

// ---------- 小工具：转义 / DOM / 按钮 / 消息 / 媒体状态 / 控件刷新 / HTTP ----------
/** 转义 HTML 特殊字符，避免商品名里的 < 破坏页面；渲染任何外部文本前都先过一遍。 */
function esc(value) {
  return String(value).replace(/[&<>"]/g,
    (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[char]));
}
/** 创建一个带类名和文本的元素；拼卡片、消息气泡这类小节点时用。 */
function el(tag, cls, text) {
  const node = document.createElement(tag); if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text; return node;
}
/** 生成一个按钮并绑定点击回调；反馈、比较、导出等按钮都走这里。 */
function btn(text, cls, onClick) {
  const node = el('button', cls, text); node.type = 'button';
  node.onclick = onClick; return node;
}
/** 往对话区追加一条消息并滚到底部；用户输入和助手回复都这样显示。 */
function chat(text, user = false) {
  const box = el('div', user ? 'user-message' : 'assistant-message'); if (!user) box.append(el('span', 'speaker', '拾物助手'));
  box.append(el('p', '', text)); $('messages').append(box);
  $('messages').scrollTop = $('messages').scrollHeight;
}
/** 更新媒体状态栏文字，error=true 时标红；录音或拍照失败时提示原因。 */
function st(text, error = false) {
  $('media-status').hidden = false; $('media-status').textContent = text;
  $('media-status').className = 'media-status' + (error ? ' error' : '');
}
/** 切换全局忙碌状态并刷新控件；请求进行中时禁用按钮，避免重复提交。 */
function busy(value) {
  state.busy = value; ctl();
}
/** 按当前状态统一刷新所有按钮的可用性；录音中、请求中或选中不足两件时该灰就灰。 */
function ctl() {
  document.querySelectorAll('button').forEach((node) => {
    node.disabled = state.busy;
  }); $('submit').textContent = state.busy ? '正在拾选…' : '发现好物 ↗';
  $('compare').disabled = state.busy || state.selected.size < 2; $('export').disabled = state.busy || !state.response;
  $('recommendation-user').disabled = state.busy || !!recording || recordingStarting; $('stop-record').disabled = !recording;
  $('record').disabled = state.busy || !!recording || recordingStarting; $('camera-stop').disabled = !camera;
  $('camera-start').disabled = state.busy || !!camera || cameraStarting; $('camera-identify').disabled = state.busy || !camera;
  $('audio-file').disabled = state.busy || !!recording; $('image-file').disabled = state.busy;
  if (!recording) return; $('submit').disabled = true;
  document.querySelectorAll('[data-mode]').forEach((node) => {
    node.disabled = true;
  });
}
/** 发一次 JSON POST 并解析响应；/api/events 和 /api/media/* 共用。 */
async function post(url, payload) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)});
  return {ok: r.ok, data: await r.json()};
}

// ---------- 事件请求与推荐结果渲染 ----------
/** 组装事件并发往 /api/events，把结果渲染成卡片；文字、找相似、撤销、比较都走它，失败时在状态栏提示重试。 */
async function send(payload, source = 'text', label) {
  if (state.busy || state.recording) return; if (label) chat(label, true);
  busy(true);
  const event = {trace_id: crypto.randomUUID(), user_id: state.user, session_id: state.session,
    timestamp: new Date().toISOString(), source, payload};
  try {
    const {data} = await post('/api/events', event); state.response = data;
    chat(data.message || '请求已完成。'); render(data);
  } catch (error) {
    chat('连接失败，请确认本地服务正在运行后重试。'); $('result-status').textContent = '本次请求没有完成。请重试。';
    $('result-status').className = 'result-status error';
  } finally { busy(false); }
}
/** 把价格格式化成 ¥/£，null 显示“价格缺失”；卡片和比较表共用。 */
function money(value, currency = 'CNY') {
  return value === null ? '价格缺失' : `${currency === 'GBP' ? '£' : '¥'}${value}`;
}
/** 把一条推荐渲染成卡片 HTML；每张卡带喜欢/不喜欢、找相似和证据折叠。 */
function cardHTML(item, index, response) {
  const preference = response.context?.preferences?.[item.item_id];
  const meta = response.dataset === 'synthetic-demo-v1' ? '合成演示商品'
    : item.price_kind === 'historical_last_observed' ? '模块一商品 · 历史单价' : '模块一商品 · 价格缺失';
  const snippets = item.evidence.snippets
    .map((snippet) => `<p>${esc(snippet.id)} — ${esc(snippet.text)}</p>`).join('');
  const feedback = (sign) => 'feedback' + (preference === sign ? ' selected' : ''); const id = esc(item.item_id);
  return `<article class="product-card">
  <div class="product-art art-${COLOR_CLASS[item.color] || 'white'}">
    <span class="product-number">NO. ${(index + 1).toString().padStart(2, '0')}</span>
    <span class="product-icon" aria-hidden="true">${ICON[item.category] || '✳'}</span>
    <label class="compare-label"><input type="checkbox" data-compare="${id}" aria-label="选择${esc(item.name)}进行比较">比较</label>
  </div>
  <div class="product-body">
    <div class="product-title-row"><h3>${esc(item.name)}</h3><span class="price">${money(item.price, item.currency)}</span></div>
    <p class="product-meta">${esc(item.color)} / ${esc(item.category)} · ${meta}</p>
    <p class="product-reason">${esc(item.explanation.text)}</p>
    <div class="product-footer">
      <button type="button" class="${feedback(1)}" data-act="like" data-id="${id}">♡ 喜欢</button>
      <button type="button" class="${feedback(-1)}" data-act="dislike" data-id="${id}">− 不喜欢</button>
      <button type="button" class="text-button similar" data-act="similar" data-id="${id}">找相似 ↗</button>
    </div>
    <details class="evidence"><summary>推荐依据 · ${item.explanation.citations.length} 项证据</summary>
      <div class="evidence-body"><p>召回来源：${esc(item.sources.join(' / '))}</p><p>排序分数：${item.score}</p>
      ${snippets}<p>分数明细：${esc(JSON.stringify(item.score_breakdown))}</p></div>
    </details>
  </div></article>`;
}
/** 用一次响应重绘整个结果区：条件标签、商品卡、手势下拉和 trace 面板；每次请求完成后调用。 */
function render(response) {
  const items = response.items || []; state.selected.clear();
  $('compare-count').textContent = '0';
  $('result-count').textContent = `/ ${items.length.toString().padStart(2, '0')}`;
  $('constraints').innerHTML = Object.entries(response.context?.constraints || {}).map(([key, value]) => {
    const unit = key.startsWith('budget_') ? (response.currency === 'GBP' ? '£' : '¥') : '';
    return `<span class="constraint">${SLOT[key] || key} · ${unit}${value}</span>`;
  }).join('') || '<span class="subtle">还没有设置筛选条件</span>'; $('result-status').textContent = response.message || '';
  $('result-status').className = 'result-status'
    + (['error', 'conflict'].includes(response.status) ? ' error' : '');
  const target = $('gesture-target'); const previous = target.value;
  const options = [['', '请选择反馈商品'], ...items.map((item) => [item.item_id, item.name])];
  target.innerHTML = options.map(([value, name]) =>
    `<option value="${esc(value)}">${esc(name)}</option>`).join('');
  if (options.some(([value]) => value === previous)) target.value = previous;
  $('products').innerHTML = items.length
    ? items.map((item, index) => cardHTML(item, index, response)).join('')
    : `<div class="empty-state"><span class="empty-art">✳</span>
      <h3>${response.status === 'empty' ? '为条件留一点余地' : '继续告诉我你的想法'}</h3>
      <p>${esc(response.message || '新的选择，从一句话开始。')}</p></div>`;
  if (response.status === 'ended') {
    state.session = crypto.randomUUID(); chat('已经开启新的会话，之前的条件与反馈已清理。');
  }
  $('trace').textContent = JSON.stringify(response, null, 2); $('trace-label').textContent = response.trace_id || '无追踪编号';
  if (response.mode) {
    $('health').textContent = response.mode === 'hybrid' ? 'Oblivionis v2'
      : response.dataset === 'online-retail-module-one-v2' ? '真实目录兜底'
        : response.perception?.provider === 'module_two' ? '本地识别 · 合成目录' : '合成目录演示';
  }
}
$('products').addEventListener('click', (event) => {
  const node = event.target.closest('[data-act]');
  const item = node && (state.response?.items || []).find((row) => row.item_id === node.dataset.id); if (!item) return;
  if (node.dataset.act === 'similar') {
    send({intent: 'similar', item_id: item.item_id}, 'text', `找与 ${item.name} 相似的商品`);
  } else {
    send({gesture: node.dataset.act === 'like' ? 'thumbs_up' : 'thumbs_down', item_id: item.item_id},
      'gesture', `${node.dataset.act === 'like' ? '喜欢' : '不喜欢'} ${item.name}`);
  }
});
$('products').addEventListener('change', (event) => {
  const check = event.target.closest('[data-compare]'); if (!check) return;
  if (check.checked) {
    state.selected.add(check.dataset.compare);
  } else {
    state.selected.delete(check.dataset.compare);
  }
  $('compare-count').textContent = String(state.selected.size); $('compare').disabled = state.busy || state.selected.size < 2;
});

// ---------- 文本 / 视觉标签 / 手势输入与模式切换 ----------
/** 解析“背包:0.95,红色:0.8”这类视觉标签输入；格式或置信度不合法就提示并返回 null。 */
function parseLabelInput(raw) {
  const labels = [];
  for (const entry of raw.trim().split(/[,，]/).filter(Boolean)) {
    const bits = entry.split(/[:：]/); const confidence = Number(bits[1]);
    if (bits.length !== 2 || !bits[0].trim() || !Number.isFinite(confidence)
        || confidence < 0 || confidence > 1) {
      chat('标签格式应为“背包:0.95”，置信度范围为0到1。'); return null;
    }
    labels.push({label: bits[0].trim(), confidence});
  }
  return labels;
}
$('composer').addEventListener('submit', (event) => {
  event.preventDefault(); if (state.recording || state.busy) return;
  const text = $('query').value.trim(); let payload, label = text;
  if (state.mode === 'gesture') {
    payload = {gesture: $('gesture').value}; if ($('gesture-target').value) payload.item_id = $('gesture-target').value;
    label = $('gesture').selectedOptions[0].textContent;
  } else if (state.mode === 'vision') {
    const labels = parseLabelInput($('labels').value); if (!labels) return;
    payload = {labels}; if (text) payload.text = text;
    label = `视觉标签：${$('labels').value}${text ? '；' + text : ''}`;
  } else if (state.mode === 'voice') {
    payload = {transcript: text};
  } else {
    if (!text) {
      $('query').focus(); return;
    }
    payload = {text};
  }
  send(payload, state.mode, label); $('query').value = '';
});
$('query').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); if (!state.busy) $('composer').requestSubmit();
  }
}); $('undo').addEventListener('click', () => send({action: 'undo'}, 'text', '撤销上一步条件'));
$('reset').addEventListener('click', () => send({action: 'reset'}, 'text', '开启新会话'));
$('compare').addEventListener('click',
  () => send({intent: 'compare', item_ids: [...state.selected]}, 'text', '比较所选商品'));
document.querySelectorAll('[data-prompt]').forEach((node) => node.addEventListener('click',
  () => send({text: node.dataset.prompt}, 'text', node.dataset.prompt)));
document.querySelectorAll('[data-mode]').forEach((tab) => tab.addEventListener('click', () => {
  state.mode = tab.dataset.mode; document.querySelectorAll('[data-mode]').forEach((node) => node.classList.toggle('active', node === tab));
  stopCamera(); $('vision-input').hidden = state.mode !== 'vision';
  $('gesture-input').hidden = state.mode !== 'gesture'; $('voice-media').hidden = state.mode !== 'voice';
  $('vision-media').hidden = state.mode !== 'vision'; $('camera-controls').hidden = !['vision', 'gesture'].includes(state.mode);
  $('camera-identify').hidden = state.mode !== 'vision'; $('query').hidden = $('input-label').hidden = state.mode === 'gesture';
  $('input-label').textContent = state.mode === 'voice' ? '或手动填写语音转写文本' : '描述你的需求'; $('input-hint').textContent = HINT[state.mode];
  $('camera-hint').textContent = state.mode === 'gesture'
    ? '保持手势：点赞=喜欢，倒拇指=不喜欢，V=继续发现，手掌=结束；反馈前先选商品。' : '开启摄像头后，点击拍摄按钮识别画面。';
  $('media-status').hidden = true; ctl();
}));

// ---------- 语音：录音、PCM16 编码与 WAV 上传 ----------
/** 把二进制分块转成 base64；录音 PCM 和上传的音频/图片都靠它进 JSON。 */
function bytes64(bytes) {
  let result = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    result += String.fromCharCode(...bytes.subarray(i, i + 8192));
  }
  return btoa(result);
}
/** 打开麦克风录 20 秒 PCM16；点“开始录音”时启动，超时或手动停止后自动识别。 */
async function startRecording() {
  if (state.busy || recording || recordingStarting) return; recordingStarting = true;
  ctl(); let stream, context;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true},
      video: false});
    if (state.mode !== 'voice') {
      stream.getTracks().forEach((track) => track.stop()); return;
    }
    context = new AudioContext({sampleRate: 16000}); await context.resume();
    const source = context.createMediaStreamSource(stream); const processor = context.createScriptProcessor(4096, 1, 1);
    const silent = context.createGain(); silent.gain.value = 0;
    recording = {stream, context, source, processor, silent, chunks: [], count: 0, rate: context.sampleRate}; state.recording = true;
    processor.onaudioprocess = (event) => {
      if (!recording) return; const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
      recording.chunks.push(chunk); recording.count += chunk.length;
      if (recording.count >= recording.rate * 20) finishRecording();
    }; source.connect(processor);
    processor.connect(silent); silent.connect(context.destination);
    recording.timer = setTimeout(finishRecording, 20000); ctl();
    st('正在录音，最长20秒；结束后自动识别。');
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop()); if (context) await context.close();
    recording = null; state.recording = false;
    st(`无法录音：${error.message}`, true);
  } finally {
    recordingStarting = false; ctl();
  }
}
/** 停止录音并释放麦克风、AudioContext；结束录音或页面关闭时调用。 */
async function releaseRecording() {
  const take = recording; recording = null;
  state.recording = false; if (!take) return null;
  clearTimeout(take.timer); take.processor.onaudioprocess = null;
  take.source.disconnect(); take.processor.disconnect();
  take.silent.disconnect(); take.stream.getTracks().forEach((track) => track.stop());
  await take.context.close(); return take;
}
/** 收尾录音：重采样到 16kHz/16bit 单声道后上传识别；录音太短会提示重录。 */
async function finishRecording() {
  const take = await releaseRecording(); if (!take) return;
  busy(false);
  if (take.count < take.rate * 0.1) {
    st('录音太短，请重录。', true); return;
  }
  const samples = new Float32Array(take.count); let offset = 0;
  for (const chunk of take.chunks) {
    samples.set(chunk, offset); offset += chunk.length;
  }
  const length = Math.min(320000, Math.floor(samples.length * 16000 / take.rate)); const pcm = new DataView(new ArrayBuffer(length * 2));
  // 采样率不是 16k 时按比例取最近点，保证协议仍是 16kHz/16bit/单声道。
  const gap = take.rate / 16000;
  for (let i = 0; i < length; i++) {
    const value = samples[Math.min(samples.length - 1, Math.floor(i * gap))];
    pcm.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  await mediaRequest('voice', {audio: bytes64(new Uint8Array(pcm.buffer)), format: 'pcm16'}, '录音需求（本机语音识别）');
}

// ---------- 媒体请求、摄像头拍照与手势轮询 ----------
/** 调一次 /api/media/<source> 做语音、图片或手势识别；结果写回输入框并渲染推荐，手势出错就停摄像头。 */
async function mediaRequest(source, payload, label) {
  if (state.busy) return; const gesture = source === 'gesture';
  const session = state.session; const stream = cameraStreamId;
  if (!gesture) {
    chat(label, true); busy(true);
    st('正在本机识别；首次加载模型需要稍等…');
  }
  try {
    const {ok, data} = await post(`/api/media/${source}`, {trace_id: crypto.randomUUID(), user_id: state.user,
      session_id: session, ...payload});
    if (session !== state.session || (gesture && stream !== cameraStreamId)) return;
    if (!ok || data.status === 'error') throw new Error(data.message || '识别失败');
    if (data.status === 'pending') {
      st(data.perception?.hand_detected ? '检测到手部，请保持手势并在两次动作之间放下手。' : '等待手势：请把手放入画面。'); return;
    }
    if (gesture && state.busy) return; // 手动请求进行中时暂停手势结果
    state.response = data;
    if (data.perception?.transcript) {
      $('query').value = data.perception.transcript;
      chat(`识别到：${data.perception.transcript}`);
    }
    if (data.perception?.labels) {
      $('labels').value = data.perception.labels.map((tag) => `${tag.label}:${tag.confidence}`).join(',');
      const names = data.perception.labels.map((tag) => `${tag.label} (${tag.confidence})`).join('、');
      chat(`识别标签：${names || '无法确认品类与颜色'}`);
    }
    if (gesture) chat(`已确认手势：${data.perception.gesture}`);
    chat(data.message || '识别完成'); render(data);
    st('本地识别完成；详情已加入本次请求证据。'); if (data.status === 'ended') stopCamera();
  } catch (error) {
    st(`${error.message}。可使用下方手动输入重试。`, true);
    if (gesture) {
      stopCamera();
    } else {
      chat(`识别未完成：${error.message}`);
    }
  } finally {
    if (!gesture) {
      busy(false);
    } else {
      ctl();
    }
  }
}
/** 截当前摄像头画面并缩到 640px 内的 JPEG base64；拍照识别和手势轮询都用它。 */
function frame64() {
  const video = $('camera-preview'); if (!video.videoWidth) throw new Error('摄像头画面尚未准备好');
  const canvas = document.createElement('canvas'); const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height); return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}
/** 每 100ms 抓一帧做手势识别；手势模式下持续轮询，切走模式或关摄像头就停。 */
async function gestureLoop() {
  if (!camera || state.mode !== 'gesture') return;
  if (!state.busy && !recording) {
    try {
      await mediaRequest('gesture', {image: frame64(), stream_id: cameraStreamId,
        item_id: $('gesture-target').value || undefined}, '摄像头手势');
    } catch (error) {
      st(error.message, true);
    }
  }
  if (camera && state.mode === 'gesture') cameraTimer = setTimeout(gestureLoop, 100);
}
/** 打开摄像头并显示预览；视觉或手势模式下点“开启摄像头”时调用。 */
async function startCamera() {
  if (state.busy || camera || cameraStarting) return; const mode = state.mode;
  cameraStarting = true; ctl();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640},
      height: {ideal: 480}}, audio: false});
    if (mode !== state.mode) {
      stream.getTracks().forEach((track) => track.stop()); return;
    }
    camera = stream; cameraStreamId = crypto.randomUUID();
    $('camera-preview').srcObject = camera; $('camera-preview').hidden = false;
    await $('camera-preview').play(); ctl();
    st('摄像头已开启，点击关闭即可停止采集。'); if (state.mode === 'gesture') gestureLoop();
  } catch (error) {
    stopCamera();
    st(`无法开启摄像头：${error.message}`, true);
  } finally {
    cameraStarting = false; ctl();
  }
}
/** 关闭摄像头并清掉轮询定时器；切换模式、结束会话或页面隐藏时调用。 */
function stopCamera() {
  clearTimeout(cameraTimer); camera?.getTracks().forEach((track) => track.stop());
  camera = null; cameraStreamId = null;
  $('camera-preview').srcObject = null; $('camera-preview').hidden = true;
  ctl();
}

// ---------- 媒体控件、证据导出、健康状态与用户切换 ----------
$('record').addEventListener('click', startRecording); $('stop-record').addEventListener('click', finishRecording);
$('audio-file').addEventListener('change', async () => {
  const file = $('audio-file').files[0]; if (!file) return;
  try {
    if (file.size > 700000) throw new Error('音频文件超过大小限制');
    await mediaRequest('voice', {audio: bytes64(new Uint8Array(await file.arrayBuffer())), format: 'wav'},
      `上传语音：${file.name}`);
  } catch (error) {
    st(error.message, true);
  } finally {
    $('audio-file').value = '';
  }
});
$('identify-image').addEventListener('click', async () => {
  const file = $('image-file').files[0];
  if (!file) {
    st('请先选择商品图片。', true); return;
  }
  try {
    if (file.size > 2 * 1024 * 1024) throw new Error('图片不能超过2MB');
    await mediaRequest('vision', {image: bytes64(new Uint8Array(await file.arrayBuffer())),
      text: $('query').value.trim() || undefined}, `识别图片：${file.name}`);
  } catch (error) {
    st(error.message, true);
  }
}); $('camera-start').addEventListener('click', startCamera);
$('camera-stop').addEventListener('click', () => {
  stopCamera(); st('摄像头已关闭。');
});
$('camera-identify').addEventListener('click', () => {
  try {
    mediaRequest('vision', {image: frame64(), text: $('query').value.trim() || undefined}, '识别摄像头当前画面');
  } catch (error) {
    st(error.message, true);
  }
});
$('export').addEventListener('click', () => {
  if (!state.response) return; const blob = new Blob([JSON.stringify(state.response, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob); const link = el('a');
  link.href = url;
  link.download = `trace-${state.response.trace_id}.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('recommendation-user').addEventListener('change', () => {
  stopCamera(); state.user = $('recommendation-user').value;
  state.session = crypto.randomUUID(); state.response = null;
  state.selected.clear(); render({status: 'ok', items: [], message: '已切换用户，开启独立会话。', context: {}});
  busy(false); chat('已切换模块一推荐用户；条件与反馈从新会话开始。');
});
fetch('/api/health').then((response) => response.json()).then((health) => {
  $('health').textContent = health.dataset === 'online-retail-module-one-v2' ? 'Oblivionis v2 已接入'
    : health.module_two?.configured ? '本地多模态可用' : '离线演示';
  if (!health.users?.length) return; $('user-controls').hidden = false;
  $('price-notice').textContent = health.price_notice;
  for (const user of health.users) {
    const option = el('option', '', user.name); option.value = user.user_id;
    $('recommendation-user').append(option);
  }
  state.user = health.users[0].user_id; $('query').placeholder = '比如：推荐红色杯子，预算10英镑以内';
  document.querySelector('.demo-note').textContent =
    '已接入模块一真实商品与冻结v2排序；价格是GBP历史观测单价。录音、图片与手势由模块二本机识别。';
  const prompts = ['为我推荐商品', '推荐红色杯子，预算10英镑以内', '推荐家居装饰'];
  document.querySelectorAll('[data-prompt]').forEach((node, index) => {
    node.dataset.prompt = prompts[index]; node.textContent = prompts[index] + ' ↗';
  });
}).catch(() => {
  $('health').textContent = '服务未连接';
});
window.addEventListener('pagehide', () => {
  stopCamera(); releaseRecording();
}); ctl();
