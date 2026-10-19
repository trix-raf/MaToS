import { initCamera } from './camera.js';
import { initSpeech } from './speech.js';
import { initTranscript } from './transcript.js';

const transcript = initTranscript();
initSpeech();
const camera = initCamera(transcript.setCameraState);

window.MaToS = window.MaToS || {};
window.MaToS.appendTranscript = transcript.append;
window.MaToS.getVideo = camera.getVideo;
