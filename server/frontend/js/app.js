import { initCamera } from './camera.js';
import { initTranscript } from './transcript.js';

const transcript = initTranscript();
const camera = initCamera(transcript.setCameraState);

window.MaToS = window.MaToS || {};
window.MaToS.appendTranscript = transcript.append;
window.MaToS.getVideo = camera.getVideo;
