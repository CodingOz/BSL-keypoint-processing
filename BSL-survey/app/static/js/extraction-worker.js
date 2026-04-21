async function extractV2(frames, frameTimestamps, fps) {
  const { FilesetResolver, HandLandmarker } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  );

  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'CPU',
    },
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    runningMode: 'VIDEO',
  });

  const framesOut = [];
  const anomalousHands = [];

  for (let i = 0; i < frames.length; i++) {
    const imageBitmap = frames[i];
    const timestampMs = Math.round(frameTimestamps[i] / 1000);
    const result = handLandmarker.detectForVideo(imageBitmap, timestampMs);

    const handsData = { left: [], right: [] };
    if (result.landmarks && result.handedness) {
      const seenLabels = new Set();
      for (let h = 0; h < result.landmarks.length; h++) {
        const label = result.handedness[h][0].categoryName.toLowerCase();
        const clusterId = label === 'left' ? 0 : 1;
        if (seenLabels.has(label)) { anomalousHands.push([i, label]); continue; }
        seenLabels.add(label);
        for (let lmIdx = 0; lmIdx < result.landmarks[h].length; lmIdx++) {
          const lm = result.landmarks[h][lmIdx];
          handsData[label].push({ cluster_id: clusterId, landmark_id: lmIdx, x: lm.x, y: lm.y, z: lm.z });
        }
      }
    }
    framesOut.push({ frame_index: i, timestamp: fps > 0 ? i / fps : null, hands: handsData });
  }

  handLandmarker.close();
  return { framesOut, anomalousHands };
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'EXTRACT') return;
  const { slotId, frames, frameTimestamps, fps, width, height, videoId, sign } = data;

  const progress = (stage, percent) =>
    self.postMessage({ type: 'PROGRESS', slotId, stage, percent });

  try {
    const metadata = {
      video_id:          videoId,
      sign:              sign,
      resolution:        [width, height],
      fps:               fps,
      frame_count:       frames.length,
      mediapipe_version: '0.10.21',
      rotation_applied:  0,
    };

    progress('extracting_V1', 0);
    const V1 = await extractV2(frames, frameTimestamps, fps);

    progress('extracting_V2', 0);
    const V2 = await extractV2(frames, frameTimestamps, fps);

    for (const frame of frames) frame.close();

    self.postMessage({
      type: 'DONE',
      slotId,
      payload: {
        V1: { metadata: { ...metadata }, frames: V1.framesOut },
        V2: { metadata: { ...metadata }, frames: V2.framesOut },
      },
    });

  } catch (err) {
    self.postMessage({ type: 'ERROR', slotId, message: err.message });
  }
};