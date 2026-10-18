/* MaToS bridge (browser side).  Injected into app.html by the FastAPI server - ./server/frontend is never edited.
 *
 *   camera.js --(HandTracker.detect)--> [tap] --> raw MediaPipe landmarks --> server (matos decoder)
 *   server --> {type:"text"} --> window.MaToS.appendTranscript()   (the hook app.js already exposes)
 *
 * The tap: this file imports the SAME module instance of /js/handtracker.js that camera.js uses, and wraps
 * HandTracker.prototype.detect so that the result of MediaPipe's detectForVideo() (raw, unmirrored, with z - what
 * the model was trained on; the drawing code's pts are smoothed, mirrored and have no z) is also sent to the server.
 * No second landmarker is created: the phone runs MediaPipe exactly once.
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
    <span class="mt-hud__link" data-k="link"><i></i><b>Connecting</b></span>
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
function renderState(s) {
  el.letter.textContent = s.hand && s.top ? s.top : '-';
  el.conf.textContent = s.hand ? `${Math.round(s.conf * 100)}%` : '';
  el.bar.style.width = `${Math.round((s.progress || 0) * 100)}%`;
  el.word.textContent = s.word || '';
  hud.classList.toggle('is-hand', !!s.hand);
}
function renderPending() { el.pending.textContent = pending.length ? pending.join(' ') + ' ' : ''; }

/* ------------------------------------------------------------------ server messages */
function onServerMsg(m) {
  switch (m.type) {
    case 'ready': setLink(link.mode === 'http' ? 'http' : 'online'); break;
    case 'state': renderState(m); break;
    case 'word': pending.push(m.text); renderPending(); break;
    case 'sentence': pending = []; renderPending(); break;
    case 'text':
      if (m.kind === 'word') { pending = []; renderPending(); }
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

/* ------------------------------------------------------------------ the tap */
function onFrame(raw, W, H) {
  if (!session) return;
  const hands = (raw.landmarks || []).map(lm => {
    const a = new Array(lm.length * 3);
    for (let i = 0; i < lm.length; i++) { a[3 * i] = round5(lm[i].x); a[3 * i + 1] = round5(lm[i].y); a[3 * i + 2] = round5(lm[i].z); }
    return a;
  });
  link.send({ type: 'f', t: round5((performance.now() - session.t0) / 1000), w: W, h: H, hands });
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
    try { onFrame(this.__mtRaw, W, H); } catch (e) { console.error('[bridge] frame not sent', e); }
  }
  return hands;
};

MT.bridge = { link, get active() { return !!session; } };      // handy in the phone's remote debugger
link.start();
