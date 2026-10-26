// stands in for /js/handtracker.js: the same surface the bridge taps (hl.detectForVideo, lastVideoTime, detect()).
export class HandTracker {
  constructor() {
    this.lastVideoTime = -1;
    this.hl = { detectForVideo: () => ({ landmarks: globalThis.__nextHands || [] }) };
  }
  detect(video, W, H) {
    if (video.currentTime === this.lastVideoTime) return [];
    this.lastVideoTime = video.currentTime;
    return this.hl.detectForVideo({}, performance.now()).landmarks;
  }
}
