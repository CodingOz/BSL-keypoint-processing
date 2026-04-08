import { ExtractionQueue } from './ExtractionQueue.js';

const queue = new ExtractionQueue();


export function registerVideoSlot(inputElement, slotId, sign) {
  const statusEl   = document.getElementById(`status-${slotId}`);
  const progressEl = document.getElementById(`progress-${slotId}`);

  // Subscribe to state/progress updates for this slot
  queue.subscribe(slotId, (id, event) => {
    updateSlotUI(slotId, event, statusEl, progressEl);
  });

  inputElement.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Generate a submission ID from timestamp + random to match v1/v2 files in R2
    const videoId = `${slotId}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;

    // Cancel any previous extraction for this slot (user changed their mind)
    queue.cancel(slotId);

    queue.enqueue(slotId, file, videoId, sign).catch((err) => {
      console.error(`Extraction failed for slot ${slotId}:`, err);
    });
  });
}

function updateSlotUI(slotId, event, statusEl, progressEl) {
  const labels = {
    idle:         '—',
    decoding:     'Decoding video…',
    extracting_v1: 'Extracting keypoints (v1)…',
    extracting_v2: 'Extracting keypoints (v2)…',
    ready:        '✓ Ready',
    failed:       '✗ Failed — please re-upload',
  };

  if (event.type === 'STATE_CHANGE') {
    if (statusEl) statusEl.textContent = labels[event.state] ?? event.state;
    if (event.state === 'ready' && progressEl) progressEl.value = 100;
  }

  if (event.type === 'PROGRESS') {
    const stage = event.stage.replace('_', ' ');
    if (statusEl)   statusEl.textContent = `${stage}… ${event.percent}%`;
    if (progressEl) progressEl.value     = event.percent;
  }
}


export function registerSubmitHandler(formElement, slotIds) {
  formElement.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Check all slots are ready before doing anything
    const notReady = slotIds.filter(
      (id) => queue.getState(id) !== ExtractionQueue.READY,
    );

    if (notReady.length > 0) {
      const still = notReady.filter(
        (id) => queue.getState(id) === ExtractionQueue.EXTRACTING,
      );
      const failed = notReady.filter(
        (id) => queue.getState(id) === ExtractionQueue.FAILED,
      );

      if (still.length > 0) {
        showMessage('Still processing your video — almost done…');
        await Promise.all(still.map((id) => waitForReady(id)));
      }

      if (failed.length > 0) {
        showMessage(`Some videos failed to process. Please re-upload: ${failed.join(', ')}`);
        return;
      }
    }

    // All slots ready — collect results and POST
    const submissions = {};
    for (const slotId of slotIds) {
      submissions[slotId] = queue.getResult(slotId);
    }

    await submitToCloudflare(submissions, formElement);
  });
}

function waitForReady(slotId) {
  return new Promise((resolve, reject) => {
    const unsub = queue.subscribe(slotId, (id, event) => {
      if (event.type === 'STATE_CHANGE') {
        if (event.state === ExtractionQueue.READY)  { unsub(); resolve(); }
        if (event.state === ExtractionQueue.FAILED) { unsub(); reject(new Error(`Slot ${slotId} failed`)); }
      }
    });

    // Guard: already done by the time we subscribe
    const current = queue.getState(slotId);
    if (current === ExtractionQueue.READY)  { unsub(); resolve(); }
    if (current === ExtractionQueue.FAILED) { unsub(); reject(new Error(`Slot ${slotId} failed`)); }
  });
}

async function submitToCloudflare(submissions, formElement) {
  const submitBtn = formElement.querySelector('[type=submit]');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Submitting…';

  try {
    const res = await fetch('https://your-worker.workers.dev/submit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(submissions),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const { id } = await res.json();
    showMessage(`Submitted successfully. Reference: ${id}`);
    formElement.reset();

  } catch (err) {
    showMessage(`Submission failed: ${err.message}`);
    submitBtn.disabled = false;
    submitBtn.textContent = 'Submit';
  }
}

function showMessage(text) {
  document.getElementById('form-message').textContent = text;
}