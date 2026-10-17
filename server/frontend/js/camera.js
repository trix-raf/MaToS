import { HandTracker, HAND_CONNECTIONS } from './handtracker.js';

const $ = id => document.getElementById(id);
const stage = $('camera-stage');
const video = $('camera-video');
const canvas = $('camera-landmarks');
const ctx = canvas.getContext('2d');
const tracking = $('camera-tracking');
const startBtn = $('camera-start');
const stopBtn = $('camera-stop');
const note = $('camera-note');
const device = $('camera-device');
const mirror = $('mirror-toggle');
const status = $('camera-status');
const statusLabel = $('camera-status-label');
const idleHeading = stage.querySelector('.camera__heading');
const idleExplain = stage.querySelector('.camera__explain');
const idleHeadingText = idleHeading.textContent;
const idleExplainText = idleExplain.textContent;

let stream = null;
let tracker = null;
let frame = 0;
let statsStart = 0;
let statsFrames = 0;
let statsMs = 0;
let statsText = '-- FPS · -- ms';
let onStateChange = () => {};

function setStatus(text, state) {
  statusLabel.textContent = text;
  status.classList.toggle('is-live', state === 'live');
  status.classList.toggle('is-error', state === 'error');
}

function drawHand(hand, width) {
  const scale = Math.min(1.6, Math.max(0.8, Math.min(width, canvas.height) / 720));
  const point = p => ({ x: mirror.checked ? p.x : width - p.x, y: p.y });
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(110, 231, 211, .9)';
  ctx.lineWidth = 3 * scale;
  for (const [a, b] of HAND_CONNECTIONS) {
    const p = point(hand.pts[a]), q = point(hand.pts[b]);
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
  }
  ctx.stroke();
  ctx.fillStyle = '#8b7cff';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2 * scale;
  for (const original of hand.pts) {
    const p = point(original);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function requestFrame() {
  return video.requestVideoFrameCallback
    ? video.requestVideoFrameCallback(drawFrame)
    : requestAnimationFrame(drawFrame);
}

function drawFrame() {
  if (!stream) return;
  if (video.readyState >= 2 && video.videoWidth) {
    const { videoWidth: width, videoHeight: height } = video;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.save();
    if (mirror.checked) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, width, height);
    ctx.restore();

    if (tracker) {
      const before = tracker.lastVideoTime;
      const started = performance.now();
      const hands = tracker.detect(video, width, height);
      if (tracker.lastVideoTime !== before) {
        statsFrames++;
        statsMs += performance.now() - started;
      }
      const now = performance.now();
      if (now - statsStart >= 500) {
        statsText = `${Math.round(statsFrames * 1000 / (now - statsStart))} FPS · ${statsFrames ? Math.round(statsMs / statsFrames) : 0} ms`;
        statsFrames = statsMs = 0;
        statsStart = now;
      }
      for (const hand of hands) drawHand(hand, width);
      tracking.textContent = `${hands.length ? `${hands.length} ${hands.length === 1 ? 'hand' : 'hands'} detected` : 'Show your hands'} · ${statsText}`;
    }
  }
  frame = requestFrame();
}

function stopCamera() {
  if (frame) {
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frame);
    else cancelAnimationFrame(frame);
  }
  frame = 0;
  if (stream) stream.getTracks().forEach(track => track.stop());
  if (tracker) tracker.reset();
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  stage.classList.remove('is-live');
  idleHeading.textContent = idleHeadingText;
  idleExplain.textContent = idleExplainText;
  stage.classList.remove('is-preparing');
  tracking.textContent = 'Show your hands';
  statsStart = statsFrames = statsMs = 0;
  statsText = '-- FPS · -- ms';
  setStatus('Camera off');
  onStateChange(false);
}

function errorMessage(err) {
  if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
    return 'Camera access was blocked. Allow it in your browser settings and try again.';
  }
  if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
    return 'No camera found. Connect one and try again.';
  }
  if (err.name === 'NotReadableError') return 'Your camera is in use by another app.';
  return err.message || 'Could not start the camera.';
}

async function loadDevices() {
  try {
    const cameras = (await navigator.mediaDevices.enumerateDevices())
      .filter(item => item.kind === 'videoinput');
    const selected = device.value;
    device.replaceChildren(new Option('Default camera', ''));
    cameras.forEach((camera, index) => {
      device.add(new Option(camera.label || `Camera ${index + 1}`, camera.deviceId));
    });
    device.value = selected;
  } catch (err) {
    console.error('Could not list cameras:', err);
  }
}

async function startCamera() {
  startBtn.disabled = true;
  stopCamera();
  stage.classList.add('is-preparing');
  idleHeading.textContent = 'Preparing for the camera…';
  idleExplain.textContent = 'Starting the camera and hand tracking. This may take a moment.';
  setNote('Preparing for the camera…');
  setStatus('Preparing for the camera…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(window.location.protocol === 'file:'
        ? 'This browser does not allow camera access from a local file.'
        : 'This browser does not support camera access.');
    }
    if (!tracker) tracker = await HandTracker.create();
    const constraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 60 },
      ...(device.value ? { deviceId: { exact: device.value } } : { facingMode: 'user' })
    };
    stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
    video.srcObject = stream;
    video.classList.toggle('is-mirrored', mirror.checked);
    await video.play();
    stage.classList.remove('is-preparing');
    stage.classList.add('is-live');
    setStatus('Camera on', 'live');
    setNote('Your browser will ask for permission');
    onStateChange(true);
    statsStart = performance.now();
    frame = requestFrame();
    loadDevices();
    if (window.MaToS?.onCameraReady) window.MaToS.onCameraReady(video);
  } catch (err) {
    stopCamera();
    setNote(errorMessage(err), true);
    console.error('Could not start MaToS camera:', err);
  } finally {
    startBtn.disabled = false;
  }
}

function setNote(text, error = false) {
  note.querySelector('span').textContent = text;
  note.classList.toggle('is-error', error);
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
device.addEventListener('change', () => { if (stream) startCamera(); });
mirror.addEventListener('change', () => {
  video.classList.toggle('is-mirrored', mirror.checked);
  if (tracker) tracker.reset();
});
window.addEventListener('pagehide', stopCamera);
loadDevices();

export function initCamera(stateChange) {
  onStateChange = stateChange;
  return { getVideo: () => video };
}
