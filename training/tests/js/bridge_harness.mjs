// Runs app/static/bridge.js (import path rewritten by the test) against a mocked browser and prints one JSON result line.
const SC = process.env.SCENARIO;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const warnings = [];
console.warn = (...a) => warnings.push(a.join(' '));

class ClassList {
  constructor() { this.s = new Set(); }
  toggle(c, on) { (on === undefined ? !this.s.has(c) : on) ? this.s.add(c) : this.s.delete(c); }
  add(c) { this.s.add(c); } remove(c) { this.s.delete(c); } contains(c) { return this.s.has(c); }
}
class El {
  constructor(tag = 'div') { this.tag = tag; this.dataset = {}; this.classList = new ClassList(); this.hidden = false; this.textContent = '';
    this.className = ''; this.style = {}; this._kids = []; this._l = {}; this.parentElement = null; }
  set innerHTML(html) {
    this._html = html; this._kids = [];
    const re = /<(\w+)([^>]*)>/g; let m;
    while ((m = re.exec(html))) {
      const k = /data-k="([^"]+)"/.exec(m[2]), mm = /data-m="([^"]+)"/.exec(m[2]);
      if (k || mm) { const e = new El(m[1]); if (k) e.dataset.k = k[1]; if (mm) e.dataset.m = mm[1]; if (/\bhidden\b/.test(m[2])) e.hidden = true; this._kids.push(e); }
    }
    const modeEl = this._kids.find(x => x.dataset.k === 'mode');
    if (modeEl) modeEl._buttons = this._kids.filter(x => x.tag === 'button');
  }
  get innerHTML() { return this._html; }
  querySelectorAll(sel) { return sel === '[data-k]' ? this._kids.filter(x => x.dataset.k) : sel === 'button' ? (this._buttons || []) : []; }
  querySelector() { return (this._b = this._b || new El('b')); }
  addEventListener(ev, fn) { (this._l[ev] = this._l[ev] || []).push(fn); }
  fire(ev, evt) { (this._l[ev] || []).forEach(f => f(evt)); }
  insertAdjacentElement(_pos, e) { globalThis.__hud = e; }
  closest() { return null; }
}
let draws = 0;
const mkCanvas = () => ({ width: 0, height: 0, getContext: () => ({ drawImage: () => { draws++; } }) });
const stage = new El(); stage.parentElement = new El();
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.document = { getElementById: id => (id === 'camera-stage' ? stage : null), createElement: t => (t === 'canvas' ? mkCanvas() : new El(t)) };
globalThis.MutationObserver = class { observe() {} };
globalThis.location = { search: process.env.TEST_QUERY || '', protocol: 'http:', host: 'localhost:8000' };
globalThis.__sockets = [];
globalThis.WebSocket = class {
  static OPEN = 1;
  constructor(url) { this.url = url; this.readyState = 0; this.sent = []; this.bufferedAmount = 0; globalThis.__sockets.push(this); }
  send(s) { this.sent.push(JSON.parse(s)); }
  close() { this.readyState = 3; }
};

await import('./bridge.mjs');
const { HandTracker } = await import('./handtracker_stub.mjs');
const MT = window.MaToS, ws = globalThis.__sockets[0], hud = globalThis.__hud;
const el = Object.fromEntries(hud.querySelectorAll('[data-k]').map(n => [n.dataset.k, n]));
ws.readyState = 1; ws.onopen();
MT.onCameraReady();                                                    // the camera is playing: a session begins
const tracker = new HandTracker(), video = { currentTime: 0 };
const hand = () => [Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + i * 0.0123456789, y: 0.4, z: 0.01 * i }))];
const frame = (hands = hand()) => { globalThis.__nextHands = hands; video.currentTime += 1 / 30; tracker.detect(video, 640, 480); };
const server = m => ws.onmessage({ data: JSON.stringify(m) });
const frames = () => ws.sent.filter(m => m.type === 'f');
const out = { warnings };
const pose = MT.bridge.pose;
const fakeLm = (opts = {}) => {
  const calls = [];
  return { calls, detectForVideo: (canvas, ts) => {
    calls.push({ ts, w: canvas.width, h: canvas.height });
    if (opts.busy) { const t = performance.now(); while (performance.now() - t < opts.busy); }
    if (opts.throws) throw new Error('model exploded');
    const n = opts.points ?? 33;
    return { landmarks: opts.none ? [] : [Array.from({ length: n }, (_, i) => ({ x: 0.3 + i / 100, y: 0.2 + i / 200, z: 0, visibility: i % 5 === 0 ? undefined : 0.9876543 }))] };
  } };
};

if (SC === 'letters_only') {
  server({ type: 'ready', bundle: 'x', words: false });
  for (let i = 0; i < 4; i++) { frame(); await sleep(5); }
  out.sent_types = ws.sent.map(m => m.type);
  out.n_frames = frames().length;
  out.any_pose = frames().some(m => 'pose' in m);
  out.hand_len = frames()[0].hands[0].length;
  out.first_hand = frames()[0].hands[0].slice(0, 6);
  out.keys = Object.keys(frames()[0]).sort();
  out.mode_hidden = el.mode.hidden;
  out.pose_state = { loading: pose.loading, failed: pose.failed, lm: !!pose.lm };
} else if (SC === 'words_pose_unavailable') {
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  await sleep(150);
  for (let i = 0; i < 3; i++) { frame(); await sleep(5); }
  out.mode_hidden = el.mode.hidden;
  out.n_frames = frames().length;
  out.any_pose = frames().some(m => 'pose' in m);
  out.pose_state = { loading: pose.loading, failed: pose.failed, lm: !!pose.lm };
} else if (SC === 'pose_off_flag') {
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  await sleep(100);
  frame();
  out.pose_state = { loading: pose.loading, failed: pose.failed, lm: !!pose.lm, off: pose.off };
  out.warned = warnings.length;
  out.mode_hidden = el.mode.hidden;
} else if (SC === 'pose_ok') {
  const lm = fakeLm(); pose.lm = lm;
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  frame();
  const f0 = frames()[0];
  out.first = { len: f0.pose && f0.pose.length, x0: f0.pose && f0.pose[0], y0: f0.pose && f0.pose[1], v0: f0.pose && f0.pose[2], v1: f0.pose && f0.pose[5], x1: f0.pose && f0.pose[3] };
  out.input = { w: lm.calls[0].w, h: lm.calls[0].h };
  for (let i = 0; i < 5; i++) frame();                                  // immediately after: not due yet
  out.burst_poses = frames().slice(1).filter(m => 'pose' in m).length;
  await sleep(90); frame();
  out.after_wait = 'pose' in frames().at(-1);
  await sleep(90); frame();
  const ts = lm.calls.map(c => c.ts);
  out.ts_increasing = ts.every((t, i) => i === 0 || t > ts[i - 1]);
  out.calls = lm.calls.length;
  out.hands_still_there = frames().every(m => m.hands.length === 1 && m.hands[0].length === 63);
} else if (SC === 'pose_slow_throttles') {
  const lm = fakeLm({ busy: 40 }); pose.lm = lm;
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  frame();
  out.every_ms = pose.everyMs;
} else if (SC === 'pose_none_or_short') {
  let lm = fakeLm({ none: true }); pose.lm = lm;
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  frame(); await sleep(80);
  lm.opts = null; pose.lm = fakeLm({ points: 20 }); frame();
  out.any_pose = frames().some(m => 'pose' in m);
  out.n_frames = frames().length;
} else if (SC === 'pose_throws') {
  const lm = fakeLm({ throws: true }); pose.lm = lm;
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 5 });
  for (let i = 0; i < 40; i++) { frame(); await sleep(8); }
  out.calls = lm.calls.length;
  out.failed = pose.failed;
  out.lm_dropped = pose.lm === null;
  out.n_frames = frames().length;
  out.any_pose = frames().some(m => 'pose' in m);
  out.gave_up_warning = warnings.some(w => w.includes('keeps failing'));
} else if (SC === 'server_messages') {
  const appended = [];
  MT.appendTranscript = t => appended.push(t);
  server({ type: 'ready', words: true, mode: 'words', pose_ms: 66 });
  pose.failed = true;                                                   // keep the test about the UI
  out.mode_visible = !el.mode.hidden;
  const on = () => el.mode.querySelectorAll('button').filter(b => b.classList.contains('is-on')).map(b => b.dataset.m);
  out.on_after_ready = on();
  server({ type: 'sign', text: 'THANK YOU', conf: 0.9 });
  out.sign = { text: el.sign.textContent, kind: el.sign.dataset.kind, pending: el.pending.textContent };
  server({ type: 'sign_miss', text: 'MOM', conf: 0.4 });
  out.miss = { text: el.sign.textContent, kind: el.sign.dataset.kind };
  server({ type: 'word', text: 'LOKY' });
  out.pending_mixed = el.pending.textContent;
  server({ type: 'text', text: 'THANK YOU', kind: 'sign' });
  out.pending_after_text = el.pending.textContent;
  server({ type: 'mode', mode: 'letters' });
  out.on_after_mode = on();
  server({ type: 'state', hand: true, top: 'A', conf: 0.8, progress: 0.5, word: 'AB', seg: true, mode: 'auto' });
  out.seg_class = hud.classList.contains('is-seg');
  server({ type: 'state', hand: false, top: null, conf: 1, progress: 0, word: '', seg: false });
  out.seg_cleared = !hud.classList.contains('is-seg');
  out.letter_shown = el.letter.textContent;
  server({ type: 'sentence', text: 'X', raw: 'X' });
  out.pending_after_sentence = el.pending.textContent;
  out.appended = appended;
  server({ type: 'totally unknown' });                                   // must be ignored
  out.survived_unknown = true;
} else if (SC === 'mode_click') {
  server({ type: 'ready', words: true, mode: 'auto', pose_ms: 66 });
  pose.failed = true;
  const btn = el.mode.querySelectorAll('button').find(b => b.dataset.m === 'words');
  el.mode.fire('click', { target: btn });
  el.mode.fire('click', { target: { dataset: {} } });                    // a click beside the buttons: nothing
  out.sent_modes = ws.sent.filter(m => m.type === 'mode');
} else if (SC === 'session_lifecycle') {
  server({ type: 'ready', words: false });
  frame();
  stage.classList.add('is-live');
  out.before = frames().length;
  MT.onCameraReady();
  out.start_sent = ws.sent.filter(m => m.type === 'start').length;
}
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
