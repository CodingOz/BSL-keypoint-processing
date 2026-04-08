import { ExtractionQueue } from './ExtractionQueue.js';

const LETTERS  = window.BSL_LETTERS;
const WORKER_URL = window.CLOUDFLARE_WORKER_URL;


const queue = new ExtractionQueue();
let readyCount = 0;

for (const letter of LETTERS) {
  const input      = document.getElementById(`field-${letter}`);
  const filenameEl = document.getElementById(`filename-${letter}`);
  const iconEl     = document.getElementById(`status-icon-${letter}`);
  const textEl     = document.getElementById(`status-text-${letter}`);
  const progressEl = document.getElementById(`progress-${letter}`);

  // Subscribe to queue events for this slot before registering the input
  queue.subscribe(letter, (_slotId, event) => {

    if (event.type === 'STATE_CHANGE') {
      updateSlotState(event.state, iconEl, textEl, progressEl);
      updateReadinessBar();
    }

    if (event.type === 'PROGRESS') {
      const stageLabel = {
        decoding:      'Decoding…',
        extracting_v1: 'Extracting (v1)…',
        extracting_v2: 'Extracting (v2)…',
      }[event.stage] ?? event.stage;

      textEl.textContent    = `${stageLabel} ${event.percent}%`;
      progressEl.value      = event.percent;
      progressEl.style.display = 'block';
    }
  });

  input.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Show filename immediately
    filenameEl.textContent = file.name;
    filenameEl.classList.add('file-selected');

    // Cancel any previous extraction for this slot (user swapped video)
    queue.cancel(letter);

    // Start extraction immediately in the background
    const videoId = `${letter}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
    queue.enqueue(letter, file, videoId, letter).catch((err) => {
      console.error(`Extraction failed for ${letter}:`, err);
    });
  });
}


function updateSlotState(state, iconEl, textEl, progressEl) {
  const config = {
    idle:         { icon: '',  text: 'Waiting for video…',            cls: '' },
    extracting:   { icon: '...', text: 'Processing…',                  cls: 'processing' },
    ready:        { icon: '✓',  text: 'Ready to submit',              cls: 'ready' },
    failed:       { icon: 'X',  text: 'Failed — please re-upload',    cls: 'failed' },
  }[state] ?? { icon: '', text: state, cls: '' };

  iconEl.textContent = config.icon;
  textEl.textContent = config.text;

  // Reset class list and apply the new state class
  const statusEl = iconEl.closest('.extraction-status');
  statusEl.className = `extraction-status ${config.cls}`;

  // Hide progress bar once finished
  if (state === 'ready' || state === 'failed' || state === 'idle') {
    progressEl.style.display = 'none';
    progressEl.value = 0;
  }
}

function updateReadinessBar() {
  readyCount = LETTERS.filter(
    (l) => queue.getState(l) === ExtractionQueue.READY
  ).length;

  const text = document.getElementById('readiness-text');
  text.textContent = `${readyCount} / ${LETTERS.length} videos ready`;

  // Colour the bar green when all ready
  const bar = document.getElementById('readiness-bar');
  bar.classList.toggle('all-ready', readyCount === LETTERS.length);
}

document.getElementById('survey-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  // Consent validation
  const c1 = document.getElementById('consent1').checked;
  const c2 = document.getElementById('consent2').checked;
  const consentError = document.getElementById('consent-error');

  if (!c1 || !c2) {
    consentError.style.display = 'block';
    consentError.scrollIntoView({ behavior: 'smooth' });
    return;
  }
  consentError.style.display = 'none';

  const missingSlots = LETTERS.filter(
    (l) => queue.getState(l) === ExtractionQueue.IDLE
  );
  if (missingSlots.length > 0) {
    showMessage(`Please upload a video for: ${missingSlots.join(', ')}`);
    return;
  }

  // Check for failures — these need re-upload before we can proceed
  const failedSlots = LETTERS.filter(
    (l) => queue.getState(l) === ExtractionQueue.FAILED
  );
  if (failedSlots.length > 0) {
    showMessage(`These videos failed to process, please re-upload: ${failedSlots.join(', ')}`);
    return;
  }

  const extractingSlots = LETTERS.filter(
    (l) => queue.getState(l) === ExtractionQueue.EXTRACTING
  );
  if (extractingSlots.length > 0) {
    showMessage(`Almost done — waiting for ${extractingSlots.length} video(s) to finish processing…`);
    try {
      await Promise.all(extractingSlots.map(waitForReady));
    } catch {
      showMessage('One or more videos failed to process. Please re-upload them.');
      return;
    }
  }

  const submissions = {};
  for (const letter of LETTERS) {
    submissions[letter] = queue.getResult(letter);
  }

  await postToCloudflare(submissions);
});


function waitForReady(slotId) {
  return new Promise((resolve, reject) => {
    // Guard: check current state first in case it already resolved
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

    // Redirect to Flask success page on completion
    window.location.href = '/success';

  } catch (err) {
    overlay.style.display = 'none';
    submitBtn.disabled = false;
    showMessage(`Submission failed: ${err.message}. Please try again.`);
  }
}

function showMessage(text) {
  const el = document.getElementById('form-message');
  el.textContent = text;
  el.scrollIntoView({ behavior: 'smooth' });
}