
const CDN_BASES = [
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14',
  'https://unpkg.com/@mediapipe/tasks-vision@0.10.14',
];
const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const INPUT_WIDTH = 224;
const SMOOTHING = 0.72;

async function loadVision() {
  let lastErr;
  for (const base of CDN_BASES) {
    for (const url of [base, base + '/vision_bundle.mjs']) {
      try {
        return await import(url);
      } catch (e) {
        lastErr = e;
      }
    }
  }
  throw new Error('Could not load the MediaPipe tracking library (internet required): ' + (lastErr && lastErr.message));
}

export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

export class HandTracker {
  static async create() {
    const vision = await loadVision();
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM_PATH);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence: 0.4, // loose, so fast rips don't drop tracking
      minHandPresenceConfidence: 0.4,
    });
    let hl;
    try {
      hl = await vision.HandLandmarker.createFromOptions(fileset, opts('GPU'));
    } catch (e) {
      hl = await vision.HandLandmarker.createFromOptions(fileset, opts('CPU'));
    }
    return new HandTracker(hl);
  }

  constructor(hl) {
    this.hl = hl;
    this.lastVideoTime = -1;
    this.lastHands = [];
    this.input = document.createElement('canvas');
    this.inputCtx = this.input.getContext('2d', { alpha: false });
  }

  reset() {
    this.lastVideoTime = -1;
    this.lastHands = [];
  }

  // Returns [{pts:[21 px points], palm:{x,y}, fistScore:0..1}]
  // Coordinates are mirrored to match the selfie view on screen.
  detect(video, W, H) {
    if (video.readyState < 2) return this.lastHands;
    if (video.currentTime === this.lastVideoTime) return this.lastHands;
    const inputHeight = Math.round(INPUT_WIDTH * H / W);
    if (this.input.width !== INPUT_WIDTH || this.input.height !== inputHeight) {
      this.input.width = INPUT_WIDTH;
      this.input.height = inputHeight;
    }
    this.inputCtx.drawImage(video, 0, 0, INPUT_WIDTH, inputHeight);

    let res;
    try {
      res = this.hl.detectForVideo(this.input, performance.now());
    } catch (e) {
      return this.lastHands;
    }
    this.lastVideoTime = video.currentTime;

    const hands = [];
    for (const lm of res.landmarks || []) {
      const pts = lm.map(p => ({ x: (1 - p.x) * W, y: p.y * H }));
      const palm = {
        x: (pts[0].x + pts[5].x + pts[17].x) / 3,
        y: (pts[0].y + pts[5].y + pts[17].y) / 3,
      };
      // fist = fingertips pulled back near the wrist vs their PIP joints
      const tips = [8, 12, 16, 20], pips = [6, 10, 14, 18];
      let curled = 0;
      for (let i = 0; i < 4; i++) {
        const dT = (lm[tips[i]].x - lm[0].x) ** 2 + (lm[tips[i]].y - lm[0].y) ** 2;
        const dP = (lm[pips[i]].x - lm[0].x) ** 2 + (lm[pips[i]].y - lm[0].y) ** 2;
        if (dT < dP * 1.1) curled++;
      }
      hands.push({ pts, palm, fistScore: curled / 4 });
    }
    const previousHands = this.lastHands;
    const matched = new Set();
    for (const hand of hands) {
      let nearest = -1, nearestDistance = Infinity;
      previousHands.forEach((previous, index) => {
        if (matched.has(index)) return;
        const distance = Math.hypot(hand.palm.x - previous.palm.x, hand.palm.y - previous.palm.y);
        if (distance < nearestDistance) {
          nearest = index;
          nearestDistance = distance;
        }
      });
      if (nearest < 0 || nearestDistance > Math.max(W, H) * 0.25) continue;
      matched.add(nearest);
      const previous = previousHands[nearest];
      const response = Math.min(0.95, SMOOTHING + nearestDistance / Math.max(W, H) * 1.5);
      hand.pts = hand.pts.map((point, index) => ({
        x: response * point.x + (1 - response) * previous.pts[index].x,
        y: response * point.y + (1 - response) * previous.pts[index].y,
      }));
      hand.palm = {
        x: (hand.pts[0].x + hand.pts[5].x + hand.pts[17].x) / 3,
        y: (hand.pts[0].y + hand.pts[5].y + hand.pts[17].y) / 3,
      };
    }
    this.lastHands = hands;
    return hands;
  }
}
