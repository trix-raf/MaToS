const $ = id => document.getElementById(id);
const panel = $('transcript');
const text = $('transcript-text');
const wordCount = $('word-count');
const saveState = $('save-state');
const copyBtn = $('copy-btn');
const downloadBtn = $('download-btn');
const speakBtn = $('speak-btn');
const clearBtn = $('clear-btn');
const sizeSelect = $('text-size');
const targetLanguage = $('target-language');
const languageLabel = $('transcript-lang');
const status = $('text-status');
const statusLabel = $('text-status-label');

let cameraActive = false;

function setStatus(label, live = false) {
  statusLabel.textContent = label;
  status.classList.toggle('is-live', live);
  status.classList.remove('is-error');
}

function refresh() {
  const value = text.textContent.trim();
  const count = value ? value.split(/\s+/).length : 0;
  panel.classList.toggle('has-text', count > 0);
  wordCount.textContent = `${count} ${count === 1 ? 'word' : 'words'}`;
  saveState.textContent = count ? 'Ready to copy or download' : 'Nothing to save yet';
  speakBtn.disabled = !count || speakBtn.dataset.unsupported === 'true';
  copyBtn.disabled = downloadBtn.disabled = clearBtn.disabled = count === 0;
}

function appendTranscript(value) {
  if (!value) return;
  text.textContent += `${text.textContent ? ' ' : ''}${value}`;
  if (cameraActive) setStatus('Translating', true);
  refresh();
  text.parentElement.scrollTop = text.parentElement.scrollHeight;
}

function flash(button, label) {
  const span = button.querySelector('span');
  const original = span.textContent;
  span.textContent = label;
  setTimeout(() => { span.textContent = original; }, 1600);
}

async function copyTranscript() {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text.textContent);
    } else {
      const field = Object.assign(document.createElement('textarea'), { value: text.textContent });
      document.body.appendChild(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      if (!copied) throw new Error('Clipboard copy was rejected.');
    }
    flash(copyBtn, 'Copied');
  } catch (err) {
    flash(copyBtn, 'Copy failed');
    console.error('Could not copy transcript:', err);
  }
}

function downloadTranscript() {
  const url = URL.createObjectURL(new Blob([text.textContent], { type: 'text/plain;charset=utf-8' }));
  const link = Object.assign(document.createElement('a'), { href: url, download: 'matos-transcript.txt' });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

copyBtn.addEventListener('click', copyTranscript);
downloadBtn.addEventListener('click', downloadTranscript);
clearBtn.addEventListener('click', () => {
  text.textContent = '';
  setStatus(cameraActive ? 'Waiting for signs' : 'Not started', cameraActive);
  refresh();
});
sizeSelect.addEventListener('change', () => panel.setAttribute('data-size', sizeSelect.value));
targetLanguage.addEventListener('change', () => {
  languageLabel.textContent = `${targetLanguage.value} transcript`;
});

export function initTranscript() {
  refresh();
  return {
    append: appendTranscript,
    setCameraState(active) {
      cameraActive = active;
      setStatus(active ? 'Waiting for signs' : text.textContent.trim() ? 'Stopped' : 'Not started', active);
    }
  };
}
