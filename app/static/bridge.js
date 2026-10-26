/* MaToS bridge (browser side).  Injected into app.html by the FastAPI server - ./server/frontend is never edited.
 *
 *   camera.js --(HandTracker.detect)--> [tap] --> raw MediaPipe landmarks --> server (matos decoder)
 *   server --> {type:"text"} --> window.MaToS.appendTranscript()   (the hook app.js already exposes)
 *
 * The tap: this file imports the SAME module instance of /js/handtracker.js that camera.js uses, and wraps
 * HandTracker.prototype.detect so that the result of MediaPipe's detectForVideo() (raw, unmirrored, with z - what
 * the model was trained on; the drawing code's pts are smoothed, mirrored and have no z) is also sent to the server.
 * Letters: no second landmarker is created - the phone runs the hand model exactly once.
 *
 * WORD level (only when the server reports a word engine in its `ready` message): a MediaPipe PoseLandmarker (lite, same
 * tasks-vision 0.10.14 module and wasm as the hand model) runs on every ~66th ms at most, throttled further if it is slow, and its
 * upper-body points (nose ... hips, 25 x [x, y, visibility]) ride along in the frame message as `pose`.  Everything else about the
 * word engine - which hand is left/right, torso normalisation, segmentation, classification - happens on the server.  If the pose
 * model cannot load, nothing breaks: the server simply never receives a pose and stays on the letter engine.
 *   ?pose=off   never load the pose model        ?pose=cpu   force the CPU delegate (try it if the phone gets slow)
 */
import { HandTracker } from '/js/handtracker.js';

const $ = id => document.getElementById(id);
const stage = $('camera-stage');
const MT = (window.MaToS = window.MaToS || {});
const round5 = v => Math.round(v * 1e5) / 1e5;

/* ------------------------------------------------------------------ HUD (under the camera) */
const hud = document.createElement('div');
hud.className = 'mt-hud';
hud.hidden = true;
hud.innerHTML = `
  <div class="mt-hud__top">
    <span class="mt-hud__letter" data-k="letter">-</span>
    <span class="mt-hud__conf" data-k="conf"></span>
    <span class="mt-hud__sign" data-k="sign"></span>
    <span class="mt-hud__link" data-k="link"><i></i><b>Connecting</b></span>
  </div>
  <div class="mt-hud__mode" data-k="mode" hidden>
    <button type="button" data-m="auto">Words + letters</button><button type="button" data-m="letters">Letters</button><button type="button" data-m="words">Words</button>
  </div>
  <div class="mt-hud__bar"><i data-k="bar"></i></div>
  <div class="mt-hud__line"><span class="mt-hud__pending" data-k="pending"></span><span class="mt-hud__word" data-k="word"></span><span class="mt-hud__caret"></span></div>`;
(stage.closest('.camera__inset') || stage.parentElement).insertAdjacentElement('afterend', hud);
const el = Object.fromEntries([...hud.querySelectorAll('[data-k]')].map(n => [n.dataset.k, n]));

let pending = [];
const linkLabels = { connecting: 'Connecting', online: 'Connected', http: 'Connected (http)', offline: 'Server unreachable' };
function setLink(state) {
  hud.dataset.link = state;
  el.link.querySelector('b').textContent = linkLabels[state] || state;
}
let signTimer = 0;
function showSign(text, cls) {                       // the word just recognised (or "?" for a miss), shown for a moment
  el.sign.textContent = text;
  el.sign.dataset.kind = cls || 'ok';
  clearTimeout(signTimer);
  signTimer = setTimeout(() => { el.sign.textContent = ''; }, 2600);
}
function setMode(mode) { for (const b of el.mode.querySelectorAll('button')) b.classList.toggle('is-on', b.dataset.m === mode); }
el.mode.addEventListener('click', e => { const m = e.target.dataset && e.target.dataset.m; if (m) link.send({ type: 'mode', mode: m }); });

function renderState(s) {
  el.letter.textContent = s.hand && s.top ? s.top : '-';
  el.conf.textContent = s.hand && s.top ? `${Math.round(s.conf * 100)}%` : '';
  hud.classList.toggle('is-seg', !!s.seg);            // a sign is in progress (the server holds the letters back until it is over)
  el.bar.style.width = `${Math.round((s.progress || 0) * 100)}%`;
  el.word.textContent = s.word || '';
  hud.classList.toggle('is-hand', !!s.hand);
}
function renderPending() { el.pending.textContent = pending.length ? pending.join(' ') + ' ' : ''; }

/* ------------------------------------------------------------------ server messages */
function onServerMsg(m) {
  switch (m.type) {
    case 'ready':
      setLink(link.mode === 'http' ? 'http' : 'online');
      if (m.words) { el.mode.hidden = false; setMode(m.mode || 'auto'); pose.enable(m.pose_ms); }
      break;
    case 'mode': setMode(m.mode); break;
    case 'sign': pending.push(m.text); renderPending(); showSign(m.text, 'ok'); break;
    case 'sign_miss': showSign(`${m.text}?`, 'miss'); break;
    case 'state': renderState(m); break;
    case 'word': pending.push(m.text); renderPending(); break;
    case 'sentence': pending = []; renderPending(); break;
    case 'text':
      if (m.kind === 'word' || m.kind === 'sign') { pending = []; renderPending(); }
      if (typeof MT.appendTranscript === 'function') MT.appendTranscript(m.text);
      break;
  }
}

/* ------------------------------------------------------------------ transport: WebSocket, falling back to HTTP */
const link = {
  mode: new URLSearchParams(location.search).get('transport') === 'http' ? 'http' : 'ws',
  ws: null, retry: 0, backoffUntil: 0,
  queue: [], busy: false,
  sid: Math.random().toString(36).slice(2) + Date.now().toString(36),

  start() { this.mode === 'ws' ? this.connectWS() : this.startHTTP(); },

  /* send(): frames are droppable, control messages (start/stop/reset/hello) are not */
  send(msg) {
    if (this.mode === 'http') {
      if (msg.type === 'f' && this.queue.length > 300) this.queue = this.queue.filter(m => m.type !== 'f').concat(msg);
      else this.queue.push(msg);
      return;
    }
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (msg.type === 'f' && ws.bufferedAmount > 120000) return;       // wifi is behind: skip, never build up lag
      ws.send(JSON.stringify(msg));
    } else if (msg.type !== 'f') {
      this.held = msg.type === 'start' || msg.type === 'stop' ? msg : this.held;   // replay the latest start/stop on (re)connect
    }
  },

  connectWS() {
    setLink('connecting');
    const ws = (this.ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`));
    let opened = false;
    const giveUp = setTimeout(() => { if (!opened) { ws.close(); } }, 4000);
    ws.onopen = () => {
      opened = true; clearTimeout(giveUp); this.retry = 0;
      ws.send(JSON.stringify(this.held || { type: 'hello' }));
      this.held = null;
    };
    ws.onmessage = e => { try { onServerMsg(JSON.parse(e.data)); } catch (_) { /* ignore */ } };
    ws.onclose = () => {
      clearTimeout(giveUp);
      if (this.ws !== ws) return;
      if (!opened) {                       // never connected: most likely the phone rejects wss:// to a self-signed cert
        console.warn('[bridge] WebSocket failed, switching to HTTP transport');
        this.mode = 'http'; this.startHTTP(); return;
      }
      setLink('connecting');
      setTimeout(() => this.connectWS(), Math.min(5000, 500 * 2 ** this.retry++));
    };
    ws.onerror = () => {};
  },

  startHTTP() {
    this.mode = 'http';
    this.queue.unshift({ type: 'hello' });
    if (this.held) { this.queue.push(this.held); this.held = null; }
    setLink('connecting');
    setInterval(() => this.pump(), 90);
  },
  async pump() {
    if (this.busy || !this.queue.length || Date.now() < this.backoffUntil) return;
    this.busy = true;
    const batch = this.queue.splice(0);
    try {
      const r = await fetch('/api/msg', { method: 'POST', headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
                                           body: JSON.stringify({ sid: this.sid, msgs: batch }), cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      (data.msgs || []).forEach(onServerMsg);
    } catch (err) {
      setLink('offline');
      this.backoffUntil = Date.now() + 1000;
      this.queue.unshift(...batch.filter(m => m.type !== 'f'));
    } finally { this.busy = false; }
  },
};

/* ------------------------------------------------------------------ camera session */
let session = null;                                          // { t0 } while the camera is running
function beginSession() {
  session = { t0: performance.now() };
  pending = []; renderPending();
  link.send({ type: 'start' });
}
function endSession() {
  if (!session) return;
  session = null;
  link.send({ type: 'stop' });                               // server flushes the last sentence -> a final "text"
  renderState({ hand: false, progress: 0, word: '' });
}
MT.onCameraReady = beginSession;                              // camera.js calls this right after the stream is playing
new MutationObserver(() => {
  const live = stage.classList.contains('is-live');
  hud.hidden = !live;
  if (!live) endSession();
}).observe(stage, { attributes: true, attributeFilter: ['class'] });
window.addEventListener('pagehide', endSession);
$('clear-btn')?.addEventListener('click', () => { pending = []; renderPending(); link.send({ type: 'reset' }); });

/* ------------------------------------------------------------------ body pose (word level only) */
const VISION_BASES = ['https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14', 'https://unpkg.com/@mediapipe/tasks-vision@0.10.14'];
const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const POSE_INPUT_WIDTH = 320;
const POSE_POINTS = 25;                                                     // nose .. hips; the legs are not sent
const poseFlag = new URLSearchParams(location.search).get('pose');
const pose = {
  lm: null, loading: false, off: poseFlag === 'off', everyMs: 66, minEveryMs: 66, lastRun: -1e9, lastTs: 0,
  canvas: document.createElement('canvas'), failed: false, errors: 0,

  enable(everyMs) {
    if (everyMs > 0) this.minEveryMs = this.everyMs = everyMs;
    if (this.off || this.lm || this.loading || this.failed) return;
    this.loading = true;
    this.load().then(lm => { this.lm = lm; }).catch(err => {
      this.failed = true;
      console.warn('[bridge] pose model unavailable - staying on the letter engine:', err && err.message);
    }).finally(() => { this.loading = false; });
  },

  async load() {
    let vision, lastErr;
    for (const base of VISION_BASES) {
      for (const url of [base, base + '/vision_bundle.mjs']) {
        try { vision = await import(url); break; } catch (e) { lastErr = e; }   // same URLs as handtracker.js: the module is shared
      }
      if (vision) break;
    }
    if (!vision) throw lastErr || new Error('tasks-vision not loadable');
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM_PATH);
    const opts = delegate => ({ baseOptions: { modelAssetPath: POSE_MODEL, delegate }, runningMode: 'VIDEO', numPoses: 1,
                                minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
    if (poseFlag !== 'cpu') {
      try { return await vision.PoseLandmarker.createFromOptions(fileset, opts('GPU')); } catch (e) { /* CPU below */ }
    }
    return vision.PoseLandmarker.createFromOptions(fileset, opts('CPU'));
  },

  /* -> a flat [x, y, visibility] * 25 array for THIS video frame, or null (not due yet / no person / not loaded) */
  run(video, W, H) {
    const now = performance.now();
    if (!this.lm || now - this.lastRun < this.everyMs) return null;
    const ph = Math.round(POSE_INPUT_WIDTH * H / W);
    if (this.canvas.width !== POSE_INPUT_WIDTH || this.canvas.height !== ph) { this.canvas.width = POSE_INPUT_WIDTH; this.canvas.height = ph; }
    this.canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, POSE_INPUT_WIDTH, ph);
    const ts = Math.max(now, this.lastTs + 1);                              // MediaPipe wants strictly increasing timestamps
    this.lastTs = ts;
    let res;
    this.lastRun = now;                                                     // a failing model must not be retried on every frame either
    try { res = this.lm.detectForVideo(this.canvas, ts); this.errors = 0; }
    catch (e) {
      if (++this.errors >= 5) {                                             // it keeps failing: give up, the server stays on the letter engine
        console.warn('[bridge] pose model keeps failing - staying on the letter engine:', e && e.message);
        this.lm = null; this.failed = true;
      }
      return null;
    }
    const dur = performance.now() - now;
    this.everyMs = Math.max(this.minEveryMs, dur * 3);                      // the pose model may use at most ~1/3 of the time budget
    const lms = res && res.landmarks && res.landmarks[0];
    if (!lms || lms.length < POSE_POINTS) return null;
    const a = new Array(POSE_POINTS * 3);
    for (let i = 0; i < POSE_POINTS; i++) {
      a[3 * i] = round5(lms[i].x); a[3 * i + 1] = round5(lms[i].y);
      a[3 * i + 2] = Math.round((lms[i].visibility == null ? 1 : lms[i].visibility) * 1000) / 1000;
    }
    return a;
  },
};

/* ------------------------------------------------------------------ the tap */
function onFrame(raw, W, H, video) {
  if (!session) return;
  const hands = (raw.landmarks || []).map(lm => {
    const a = new Array(lm.length * 3);
    for (let i = 0; i < lm.length; i++) { a[3 * i] = round5(lm[i].x); a[3 * i + 1] = round5(lm[i].y); a[3 * i + 2] = round5(lm[i].z); }
    return a;
  });
  const msg = { type: 'f', t: round5((performance.now() - session.t0) / 1000), w: W, h: H, hands };
  const p = pose.run(video, W, H);                           // null most of the time: pose is sent at ~15 Hz, hands at the video rate
  if (p) msg.pose = p;
  link.send(msg);
}

const originalDetect = HandTracker.prototype.detect;
HandTracker.prototype.detect = function (video, W, H) {
  if (!this.__mtTapped && this.hl && typeof this.hl.detectForVideo === 'function') {
    const inner = this.hl.detectForVideo.bind(this.hl);
    this.hl.detectForVideo = (img, ts) => (this.__mtRaw = inner(img, ts));
    this.__mtTapped = true;
  }
  const before = this.lastVideoTime;
  this.__mtRaw = null;
  const hands = originalDetect.call(this, video, W, H);
  if (this.lastVideoTime !== before && this.__mtRaw) {       // a NEW video frame went through MediaPipe (also when it saw no hand)
    try { onFrame(this.__mtRaw, W, H, video); } catch (e) { console.error('[bridge] frame not sent', e); }
  }
  return hands;
};

MT.bridge = { link, pose, get active() { return !!session; } };      // handy in the phone's remote debugger
link.start();
