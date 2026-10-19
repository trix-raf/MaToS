const speakButton = document.getElementById('speak-btn');
const transcriptText = document.getElementById('transcript-text');
const targetLanguage = document.getElementById('target-language');
const buttonLabel = speakButton.querySelector('span');

const languageCodes = {
  English: 'en-US',
  'Bahasa Indonesia': 'id-ID'
};

let currentUtterance = null;

function setButtonState(label, speaking) {
  buttonLabel.textContent = label;
  speakButton.setAttribute('aria-pressed', String(speaking));
}

function stopSpeech() {
  currentUtterance = null;
  window.speechSynthesis.cancel();
  setButtonState('Listen', false);
}

function speakTranscript() {
  const value = transcriptText.textContent.trim();
  if (!value) return;

  const utterance = new SpeechSynthesisUtterance(value);
  utterance.lang = languageCodes[targetLanguage.value] || navigator.language;
  utterance.onstart = () => {
    if (currentUtterance === utterance) setButtonState('Stop', true);
  };
  utterance.onend = () => {
    if (currentUtterance === utterance) stopSpeech();
  };
  utterance.onerror = event => {
    if (currentUtterance !== utterance) return;
    currentUtterance = null;
    setButtonState('Playback failed', false);
    console.error('Could not speak transcript:', event.error);
  };

  window.speechSynthesis.cancel();
  currentUtterance = utterance;
  setButtonState('Listen', false);
  try {
    window.speechSynthesis.speak(utterance);
  } catch (error) {
    currentUtterance = null;
    setButtonState('Playback failed', false);
    console.error('Could not speak transcript:', error);
  }
}

export function initSpeech() {
  if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
    speakButton.dataset.unsupported = 'true';
    speakButton.disabled = true;
    speakButton.title = 'Speech playback is not supported by this browser.';
    return;
  }

  speakButton.addEventListener('click', () => {
    if (currentUtterance) stopSpeech();
    else speakTranscript();
  });
}
