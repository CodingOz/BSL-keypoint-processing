import { ExtractionQueue } from './ExtractionQueue.js';

const LETTERS    = window.BSL_LETTERS;
const WORKER_URL = window.CLOUDFLARE_WORKER_URL;

const queue = new ExtractionQueue();

async function extractFramesFromVideo(file, onProgress) {
  const video = document.createElement('video');
  video.muted      = true;
  video.playsInline = true;
  // Must be in DOM for seeking to work reliably in all browsers
  video.style.cssText = 'position:fixed;opacity:0;pointer-events:none;width:1px;height:1px;';
  document.body.appendChild(video);

  const url = URL.createObjectURL(file);

  try {
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = () => reject(new Error('Failed to load video metadata'));
      video.src = url;
    });

    const duration       = video.duration;
    const targetFps      = 30;
    const frameCount     = Math.floor(duration * targetFps);
    const width          = video.videoWidth;
    const height         = video.videoHeight;
    const frames         = [];
    const frameTimestamps = [];

    const canvas = document.createElement('canvas');
    canvas.width  = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    for (let i = 0; i < frameCount; i++) {
      video.currentTime = i / targetFps;
      await new Promise(r => { video.onseeked = r; });
      ctx.drawImage(video, 0, 0);
      frames.push(await createImageBitmap(canvas));
      frameTimestamps.push(Math.round((i / targetFps) * 1_000_000)); // microseconds
      onProgress?.(Math.round((i / frameCount) * 100));
    }

    return { frames, frameTimestamps, fps: targetFps, width, height };

  } finally {
    URL.revokeObjectURL(url);
    document.body.removeChild(video);
  }
}

for (const letter of LETTERS) {
  const input      = document.getElementById(`video_${letter}`);
  const filenameEl = document.getElementById(`filename-${letter}`);
  const iconEl     = document.getElementById(`status-icon-${letter}`);
  const textEl     = document.getElementById(`status-text-${letter}`);
  const progressEl = document.getElementById(`progress-${letter}`);

  queue.subscribe(letter, (_slotId, event) => {
    if (event.type === 'STATE_CHANGE') {
      updateSlotState(event.state, iconEl, textEl, progressEl);
      updateReadinessBar();
    }
    if (event.type === 'PROGRESS') {
      const stageLabel = {
        decoding:      'Decoding…',
        extracting_V1: 'Extracting (v1)…',
        extracting_V2: 'Extracting (v2)…',
      }[event.stage] ?? event.stage;
      textEl.textContent       = `${stageLabel} ${event.percent}%`;
      progressEl.value         = event.percent;
      progressEl.style.display = 'block';
    }
  });

  input.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    filenameEl.textContent = file.name;
    filenameEl.classList.add('file-selected');

    queue.cancel(letter);

    // Show decoding progress on main thread
    iconEl.textContent       = '...';
    textEl.textContent       = 'Decoding video…';
    progressEl.style.display = 'block';
    progressEl.value         = 0;

    let decodedFrames;
    try {
      decodedFrames = await extractFramesFromVideo(file, (pct) => {
        textEl.textContent = `Decoding… ${pct}%`;
        progressEl.value   = pct;
      });
    } catch (err) {
      textEl.textContent       = '✗ Failed to decode — please re-upload';
      progressEl.style.display = 'none';
      iconEl.textContent       = 'X';
      console.error('Frame extraction failed for', letter, err);
      return;
    }

    const videoId = `${letter}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
    const { frames, frameTimestamps, fps, width, height } = decodedFrames;

    queue.enqueue(letter, frames, frameTimestamps, fps, width, height, videoId, letter)
      .catch((err) => console.error(`Extraction failed for ${letter}:`, err));
  });
}

function updateSlotState(state, iconEl, textEl, progressEl) {
  const config = {
    idle:       { icon: '',    text: 'Waiting for video…',         cls: '' },
    extracting: { icon: '...', text: 'Processing…',                cls: 'processing' },
    ready:      { icon: '✓',   text: 'Ready to submit',            cls: 'ready' },
    failed:     { icon: 'X',   text: 'Failed — please re-upload',  cls: 'failed' },
  }[state] ?? { icon: '', text: state, cls: '' };

  iconEl.textContent = config.icon;
  textEl.textContent = config.text;

  const statusEl = iconEl.closest('.extraction-status');
  statusEl.className = `extraction-status ${config.cls}`;

  if (state === 'ready' || state === 'failed' || state === 'idle') {
    progressEl.style.display = 'none';
    progressEl.value = 0;
  }
}

function updateReadinessBar() {
  const readyCount = LETTERS.filter(l => queue.getState(l) === ExtractionQueue.READY).length;
  document.getElementById('readiness-text').textContent =
    `${readyCount} / ${LETTERS.length} videos ready`;
  document.getElementById('readiness-bar')
    .classList.toggle('all-ready', readyCount === LETTERS.length);
}

document.getElementById('survey-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const c1 = document.getElementById('consent_video_deletion').checked;
  const c2 = document.getElementById('consent_data_usage').checked;
  const consentError = document.getElementById('consent-error');
  if (!c1 || !c2) {
    consentError.style.display = 'block';
    consentError.scrollIntoView({ behavior: 'smooth' });
    return;
  }
  consentError.style.display = 'none';

  const missingSlots = LETTERS.filter(l => queue.getState(l) === ExtractionQueue.IDLE);
  if (missingSlots.length > 0) {
    showMessage(`Please upload a video for: ${missingSlots.join(', ')}`);
    return;
  }

  const failedSlots = LETTERS.filter(l => queue.getState(l) === ExtractionQueue.FAILED);
  if (failedSlots.length > 0) {
    showMessage(`These videos failed to process, please re-upload: ${failedSlots.join(', ')}`);
    return;
  }

  const extractingSlots = LETTERS.filter(l => queue.getState(l) === ExtractionQueue.EXTRACTING);
  if (extractingSlots.length > 0) {
    showMessage(`Almost done, waiting for ${extractingSlots.length} video(s) to finish…`);
    try {
      await Promise.all(extractingSlots.map(waitForReady));
    } catch {
      showMessage('One or more videos failed to process. Please re-upload them.');
      return;
    }
  }

  const submissions = {};
  for (const letter of LETTERS) submissions[letter] = queue.getResult(letter);
  await postToCloudflare(submissions);
});

function waitForReady(slotId) {
  return new Promise((resolve, reject) => {
    const current = queue.getState(slotId);
    if (current === ExtractionQueue.READY)  return resolve();
    if (current === ExtractionQueue.FAILED) return reject();
    const unsub = queue.subscribe(slotId, (_id, event) => {
      if (event.type !== 'STATE_CHANGE') return;
      if (event.state === ExtractionQueue.READY)  { unsub(); resolve(); }
      if (event.state === ExtractionQueue.FAILED) { unsub(); reject(); }
    });
  });
}

async function postToCloudflare(submissions) {
  const submitBtn = document.getElementById('submit-btn');
  const overlay   = document.getElementById('loading-overlay');
  submitBtn.disabled = true;
  overlay.style.display = 'flex';

  try {
    const res = await fetch(WORKER_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(submissions),
    });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    window.location.href = '/success';
  } catch (err) {
    overlay.style.display = 'none';
    submitBtn.disabled = false;
    showMessage(`Submission failed: ${err.message}. Please try again.`);
  }
}

function showMessage(text) {
  const el = document.getElementById('form-message');
  el.textContent    = text;
  el.style.display  = 'block';
  el.scrollIntoView({ behavior: 'smooth' });
}