export class ExtractionQueue {
  #workers    = [];      // Worker instances
  #busy       = [];      // boolean per worker for if processing a job
  #queue      = [];      // { slotId, videoFile, videoId, sign, resolve, reject }
  #results    = new Map();
  #states     = new Map();
  #listeners  = new Map();

  // ExtractionState values
  static IDLE       = 'idle';
  static EXTRACTING = 'extracting';
  static READY      = 'ready';
  static FAILED     = 'failed';

  constructor(workerCount = Math.min(navigator.hardwareConcurrency ?? 2, 4)) {
    for (let i = 0; i < workerCount; i++) {
      const w = new Worker(
        new URL('./extraction-worker.js', import.meta.url),
        { type: 'module' }
      );
      w.onmessage = (e) => this.#handleMessage(i, e.data);
      w.onerror   = (e) => this.#handleWorkerError(i, e);
      this.#workers.push(w);
      this.#busy.push(false);
    }
  }

  enqueue(slotId, videoFile, videoId, sign) {
    // If already have a result for this slot (user replaced video), discard old
    this.#results.delete(slotId);
    this.#setState(slotId, ExtractionQueue.EXTRACTING);

    return new Promise((resolve, reject) => {
      this.#queue.push({ slotId, videoFile, videoId, sign, resolve, reject });
      this.#dispatch();
    });
  }

  getResult(slotId) {
    return this.#results.get(slotId) ?? null;
  }

  getState(slotId) {
    return this.#states.get(slotId) ?? ExtractionQueue.IDLE;
  }

  subscribe(slotId, callback) {
    if (!this.#listeners.has(slotId)) this.#listeners.set(slotId, new Set());
    this.#listeners.get(slotId).add(callback);
    return () => this.#listeners.get(slotId).delete(callback); // unsubscribe fn
  }

  cancel(slotId) {
    this.#queue = this.#queue.filter(j => j.slotId !== slotId);
    this.#results.delete(slotId);
    this.#setState(slotId, ExtractionQueue.IDLE);
  }

  #dispatch() {
    if (this.#queue.length === 0) return;

    const freeIdx = this.#busy.findIndex(b => !b);
    if (freeIdx === -1) return;

    const job = this.#queue.shift();
    this.#busy[freeIdx] = true;
    // Tag the worker so we can match it when results arrive
    this.#workers[freeIdx]._currentSlot = job.slotId;
    this.#workers[freeIdx]._currentResolve = job.resolve;
    this.#workers[freeIdx]._currentReject  = job.reject;

    this.#workers[freeIdx].postMessage({
      type:      'EXTRACT',
      slotId:    job.slotId,
      videoFile: job.videoFile,
      videoId:   job.videoId,
      sign:      job.sign,
    });
  }

  #handleMessage(workerIdx, data) {
    const { type, slotId } = data;
    this.#emit(slotId, data);

    if (type === 'PROGRESS') {
      // Progress events don't change the overall state — just forward to UI
      return;
    }

    if (type === 'DONE') {
      this.#results.set(slotId, data.payload);
      this.#setState(slotId, ExtractionQueue.READY);
      this.#workers[workerIdx]._currentResolve?.(data.payload);
      this.#free(workerIdx);
    }

    if (type === 'ERROR') {
      this.#setState(slotId, ExtractionQueue.FAILED);
      this.#workers[workerIdx]._currentReject?.(new Error(data.message));
      this.#free(workerIdx);
    }
  }

  #handleWorkerError(workerIdx, err) {
    const slotId = this.#workers[workerIdx]._currentSlot;
    if (slotId) {
      this.#setState(slotId, ExtractionQueue.FAILED);
      this.#workers[workerIdx]._currentReject?.(err);
    }
    this.#free(workerIdx);
  }

  #free(workerIdx) {
    this.#busy[workerIdx] = false;
    this.#workers[workerIdx]._currentSlot    = null;
    this.#workers[workerIdx]._currentResolve = null;
    this.#workers[workerIdx]._currentReject  = null;
    this.#dispatch(); // pick up next queued job if any
  }

  #setState(slotId, state) {
    this.#states.set(slotId, state);
    this.#emit(slotId, { type: 'STATE_CHANGE', slotId, state });
  }

  #emit(slotId, event) {
    this.#listeners.get(slotId)?.forEach(cb => cb(slotId, event));
  }
}